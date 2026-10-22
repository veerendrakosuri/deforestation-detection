"""Generate synthetic GeoTIFFs and run the complete local workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deforestation_detection.config import load_config
from deforestation_detection.geospatial import run_change_detection, run_ndvi_baseline
from deforestation_detection.inference import predict_geotiff
from deforestation_detection.prepare import prepare_dataset
from deforestation_detection.train import train_model
from deforestation_detection.visualize import create_change_map


def _spectral_image(forest: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    forest_signature = np.array([0.05, 0.07, 0.10, 0.65, 0.20, 0.15], dtype=np.float32)
    cleared_signature = np.array([0.16, 0.19, 0.31, 0.24, 0.38, 0.34], dtype=np.float32)
    image = np.where(
        forest[None, ...],
        forest_signature[:, None, None],
        cleared_signature[:, None, None],
    )
    noise = rng.normal(0, 0.008, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def _write_multiband(path: Path, array: np.ndarray, *, descriptions: tuple[str, ...]) -> None:
    import rasterio
    from rasterio.transform import from_origin

    profile = {
        "driver": "GTiff",
        "height": array.shape[1],
        "width": array.shape[2],
        "count": array.shape[0],
        "dtype": "float32",
        "crs": "EPSG:32720",
        "transform": from_origin(500_000, 9_000_000, 10, 10),
        "nodata": -9999.0,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array.astype(np.float32))
        for index, description in enumerate(descriptions, start=1):
            dataset.set_band_description(index, description)


def _write_label(path: Path, mask: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32720",
        "transform": from_origin(500_000, 9_000_000, 10, 10),
        "nodata": 255,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(mask.astype(np.uint8), 1)
        dataset.set_band_description(1, "forest")


def generate_inputs(root: Path) -> dict[str, Path]:
    size = 192
    rows, cols = np.indices((size, size))
    before_forest = ((rows // 24 + cols // 24) % 3 != 0)
    after_forest = before_forest.copy()
    after_forest[64:128, 96:160] = False
    after_forest[12:15, 12:15] = False
    rng = np.random.default_rng(7)
    before_image = _spectral_image(before_forest, rng)
    after_image = _spectral_image(after_forest, rng)

    paths = {
        "before_imagery": root / "before_sentinel2.tif",
        "before_labels": root / "before_forest_label.tif",
        "after_imagery": root / "after_sentinel2.tif",
        "after_labels": root / "after_forest_label.tif",
    }
    bands = ("B2", "B3", "B4", "B8", "B11", "B12")
    _write_multiband(paths["before_imagery"], before_image, descriptions=bands)
    _write_multiband(paths["after_imagery"], after_image, descriptions=bands)
    _write_label(paths["before_labels"], before_forest)
    _write_label(paths["after_labels"], after_forest)
    return paths


def main() -> None:
    root = Path("outputs/smoke_test")
    inputs = generate_inputs(root / "inputs")
    config = load_config("configs/smoke_test.yaml")

    baseline_before = run_ndvi_baseline(
        inputs["before_imagery"],
        inputs["before_labels"],
        root / "baseline" / "before",
        config,
    )
    baseline_after = run_ndvi_baseline(
        inputs["after_imagery"],
        inputs["after_labels"],
        root / "baseline" / "after",
        config,
    )
    expected_change = run_change_detection(
        inputs["before_labels"],
        inputs["after_labels"],
        root / "expected_change",
        config,
    )
    dataset = prepare_dataset(
        inputs["before_imagery"],
        inputs["before_labels"],
        root / "dataset",
        config,
    )
    training = config.training
    trained = train_model(
        root / "dataset",
        root / "model",
        model_name=training.model,
        encoder=training.encoder,
        encoder_weights=training.encoder_weights,
        base_channels=training.base_channels,
        epochs=training.epochs,
        batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        prediction_threshold=training.prediction_threshold,
        seed=config.random_seed,
    )
    before_prediction = predict_geotiff(
        inputs["before_imagery"], trained.checkpoint, root / "predictions" / "before"
    )
    after_prediction = predict_geotiff(
        inputs["after_imagery"], trained.checkpoint, root / "predictions" / "after"
    )
    change = run_change_detection(
        before_prediction.mask_path,
        after_prediction.mask_path,
        root / "change",
        config,
    )
    map_path = create_change_map(
        inputs["before_imagery"],
        inputs["after_imagery"],
        change["mask_path"],
        root / "deforestation_map.html",
        band_names=config.imagery.bands,
        cleared_hectares=float(change["cleared_hectares"]),
    )
    result = {
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "expected_change": expected_change,
        "dataset": dataset.to_dict(),
        "training": trained.to_dict(),
        "before_prediction": before_prediction.to_dict(),
        "after_prediction": after_prediction.to_dict(),
        "change": change,
        "map": str(map_path),
    }
    (root / "smoke_test_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

