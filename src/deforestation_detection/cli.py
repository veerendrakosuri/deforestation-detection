"""Command-line interface for the complete deforestation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import compute_ndvi, forest_mask_from_ndvi
from .change import detect_clearing
from .metrics import binary_segmentation_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_baseline(args: argparse.Namespace) -> None:
    red = np.load(args.red)
    nir = np.load(args.nir)
    ndvi = compute_ndvi(red, nir)
    forest = forest_mask_from_ndvi(red, nir, threshold=args.threshold)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "ndvi.npy", ndvi)
    np.save(output / "forest_mask.npy", forest)

    summary: dict[str, object] = {
        "threshold": args.threshold,
        "valid_pixels": int(np.count_nonzero(np.isfinite(ndvi))),
        "forest_pixels": int(np.count_nonzero(forest)),
    }
    if args.labels is not None:
        labels = np.load(args.labels)
        valid = np.isfinite(ndvi)
        summary["metrics"] = binary_segmentation_metrics(
            forest, labels, valid_mask=valid
        ).to_dict()

    _write_json(output / "baseline_metrics.json", summary)
    print(json.dumps(summary, indent=2))


def run_change(args: argparse.Namespace) -> None:
    before = np.load(args.before)
    after = np.load(args.after)
    cleared, stats = detect_clearing(
        before,
        after,
        minimum_patch_pixels=args.minimum_patch_pixels,
        pixel_size_m=args.pixel_size_m,
        connectivity=args.connectivity,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "cleared_mask.npy", cleared)
    _write_json(output / "change_metrics.json", stats.to_dict())
    print(json.dumps(stats.to_dict(), indent=2))


def run_demo(args: argparse.Namespace) -> None:
    """Run a small deterministic example without external data."""

    size = 128
    red = np.full((size, size), 0.12, dtype=np.float32)
    nir = np.full((size, size), 0.55, dtype=np.float32)
    labels = np.ones((size, size), dtype=bool)

    red[:24, :24] = 0.25
    nir[:24, :24] = 0.20
    labels[:24, :24] = False

    before = forest_mask_from_ndvi(red, nir, threshold=0.6)
    after = before.copy()
    after[48:68, 54:79] = False
    after[90:93, 90:93] = False

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "demo_red.npy", red)
    np.save(output / "demo_nir.npy", nir)
    np.save(output / "demo_forest_before.npy", before)
    np.save(output / "demo_forest_after.npy", after)

    baseline_metrics = binary_segmentation_metrics(before, labels)
    cleared, change_stats = detect_clearing(
        before,
        after,
        minimum_patch_pixels=50,
        pixel_size_m=10,
    )
    np.save(output / "demo_cleared.npy", cleared)

    summary = {
        "baseline": baseline_metrics.to_dict(),
        "change": change_stats.to_dict(),
    }
    _write_json(output / "demo_metrics.json", summary)
    print(json.dumps(summary, indent=2))


def run_export(args: argparse.Namespace) -> None:
    from .config import load_config
    from .earth_engine import start_exports

    config = load_config(args.config)
    tasks = start_exports(
        config,
        drive_folder=args.drive_folder,
        earth_engine_project=args.earth_engine_project,
        authenticate=args.authenticate,
    )
    print(json.dumps(tasks, indent=2))


def run_prepare(args: argparse.Namespace) -> None:
    from .config import load_config
    from .prepare import prepare_dataset

    summary = prepare_dataset(
        args.imagery,
        args.labels,
        args.output,
        load_config(args.config),
    )
    print(json.dumps(summary.to_dict(), indent=2))


def _encoder_weights(value: str | None) -> str | None:
    return None if value in {None, "none", "None"} else value


def run_train(args: argparse.Namespace) -> None:
    from .config import load_config
    from .train import train_model

    config = load_config(args.config)
    training = config.training
    result = train_model(
        args.dataset,
        args.output,
        model_name=args.model or training.model,
        encoder=args.encoder or training.encoder,
        encoder_weights=_encoder_weights(
            args.encoder_weights
            if args.encoder_weights is not None
            else training.encoder_weights
        ),
        base_channels=args.base_channels or training.base_channels,
        epochs=args.epochs or training.epochs,
        batch_size=args.batch_size or training.batch_size,
        learning_rate=args.learning_rate or training.learning_rate,
        weight_decay=training.weight_decay,
        prediction_threshold=training.prediction_threshold,
        seed=config.random_seed,
        device_name=args.device,
    )
    print(json.dumps(result.to_dict(), indent=2))


def run_predict(args: argparse.Namespace) -> None:
    from .inference import predict_geotiff

    result = predict_geotiff(
        args.imagery,
        args.checkpoint,
        args.output,
        device_name=args.device,
    )
    print(json.dumps(result.to_dict(), indent=2))


def run_baseline_geotiff(args: argparse.Namespace) -> None:
    from .config import load_config
    from .geospatial import run_ndvi_baseline

    result = run_ndvi_baseline(
        args.imagery,
        args.labels,
        args.output,
        load_config(args.config),
    )
    print(json.dumps(result, indent=2))


def run_change_geotiff(args: argparse.Namespace) -> None:
    from .config import load_config
    from .geospatial import run_change_detection

    result = run_change_detection(
        args.before,
        args.after,
        args.output,
        load_config(args.config),
    )
    print(json.dumps(result, indent=2))


def run_map(args: argparse.Namespace) -> None:
    from .config import load_config
    from .visualize import create_change_map

    config = load_config(args.config)
    destination = create_change_map(
        args.before,
        args.after,
        args.clearing,
        args.output,
        band_names=config.imagery.bands,
        cleared_hectares=args.cleared_hectares,
    )
    print(json.dumps({"map": str(destination)}, indent=2))


def run_workflow(args: argparse.Namespace) -> None:
    """Run baseline, preparation, training, inference, change, and mapping."""

    from .config import load_config
    from .geospatial import run_change_detection, run_ndvi_baseline
    from .inference import predict_geotiff
    from .prepare import prepare_dataset
    from .train import train_model
    from .visualize import create_change_map

    config = load_config(args.config)
    root = Path(args.output)
    baseline_before = run_ndvi_baseline(
        args.before_imagery,
        args.before_labels,
        root / "baseline" / "before",
        config,
    )
    baseline_after = run_ndvi_baseline(
        args.after_imagery,
        args.after_labels,
        root / "baseline" / "after",
        config,
    )
    dataset_summary = prepare_dataset(
        args.before_imagery,
        args.before_labels,
        root / "dataset",
        config,
    )
    training = config.training
    training_result = train_model(
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
        device_name=args.device,
    )
    checkpoint = training_result.checkpoint
    before_prediction = predict_geotiff(
        args.before_imagery,
        checkpoint,
        root / "predictions" / "before",
        device_name=args.device,
    )
    after_prediction = predict_geotiff(
        args.after_imagery,
        checkpoint,
        root / "predictions" / "after",
        device_name=args.device,
    )
    change_result = run_change_detection(
        before_prediction.mask_path,
        after_prediction.mask_path,
        root / "change",
        config,
    )
    map_path = create_change_map(
        args.before_imagery,
        args.after_imagery,
        change_result["mask_path"],
        root / "deforestation_map.html",
        band_names=config.imagery.bands,
        cleared_hectares=float(change_result["cleared_hectares"]),
    )
    result = {
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "dataset": dataset_summary.to_dict(),
        "training": training_result.to_dict(),
        "before_prediction": before_prediction.to_dict(),
        "after_prediction": after_prediction.to_dict(),
        "change": change_result,
        "map": str(map_path),
    }
    _write_json(root / "workflow_results.json", result)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deforestation-detection",
        description="Train and run a Sentinel-2 deforestation-detection pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Create an NDVI forest mask")
    baseline.add_argument("--red", required=True, help="Path to a red-band .npy array")
    baseline.add_argument("--nir", required=True, help="Path to a NIR-band .npy array")
    baseline.add_argument("--labels", help="Optional ground-truth forest-mask .npy array")
    baseline.add_argument("--threshold", type=float, default=0.6)
    baseline.add_argument("--output", default="outputs/baseline")
    baseline.set_defaults(handler=run_baseline)

    change = subparsers.add_parser("change", help="Detect forest loss between two masks")
    change.add_argument("--before", required=True, help="Earlier forest-mask .npy array")
    change.add_argument("--after", required=True, help="Later forest-mask .npy array")
    change.add_argument("--minimum-patch-pixels", type=int, default=50)
    change.add_argument("--pixel-size-m", type=float, default=10.0)
    change.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    change.add_argument("--output", default="outputs/change")
    change.set_defaults(handler=run_change)

    demo = subparsers.add_parser("demo", help="Run the pipeline on synthetic arrays")
    demo.add_argument("--output", default="outputs/demo")
    demo.set_defaults(handler=run_demo)

    export = subparsers.add_parser(
        "export-earth-engine", help="Export imagery and labels to Google Drive"
    )
    export.add_argument("--config", default="configs/rondonia_example.yaml")
    export.add_argument("--drive-folder", default="deforestation-detection")
    export.add_argument("--earth-engine-project")
    export.add_argument("--authenticate", action="store_true")
    export.set_defaults(handler=run_export)

    prepare = subparsers.add_parser("prepare", help="Create geographically split NPZ tiles")
    prepare.add_argument("--imagery", required=True)
    prepare.add_argument("--labels", required=True)
    prepare.add_argument("--config", default="configs/rondonia_example.yaml")
    prepare.add_argument("--output", default="data/processed")
    prepare.set_defaults(handler=run_prepare)

    train = subparsers.add_parser("train", help="Train and evaluate the U-Net")
    train.add_argument("--dataset", default="data/processed")
    train.add_argument("--config", default="configs/rondonia_example.yaml")
    train.add_argument("--output", default="models/run")
    train.add_argument("--model", choices=("smp-unet", "tiny-unet"))
    train.add_argument("--encoder")
    train.add_argument("--encoder-weights")
    train.add_argument("--base-channels", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=run_train)

    predict = subparsers.add_parser("predict", help="Predict a forest-mask GeoTIFF")
    predict.add_argument("--imagery", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(handler=run_predict)

    baseline_tif = subparsers.add_parser(
        "baseline-geotiff", help="Evaluate NDVI on aligned GeoTIFFs"
    )
    baseline_tif.add_argument("--imagery", required=True)
    baseline_tif.add_argument("--labels", required=True)
    baseline_tif.add_argument("--config", default="configs/rondonia_example.yaml")
    baseline_tif.add_argument("--output", default="outputs/baseline")
    baseline_tif.set_defaults(handler=run_baseline_geotiff)

    change_tif = subparsers.add_parser(
        "change-geotiff", help="Detect change between forest-mask GeoTIFFs"
    )
    change_tif.add_argument("--before", required=True)
    change_tif.add_argument("--after", required=True)
    change_tif.add_argument("--config", default="configs/rondonia_example.yaml")
    change_tif.add_argument("--output", default="outputs/change")
    change_tif.set_defaults(handler=run_change_geotiff)

    map_command = subparsers.add_parser("map", help="Create an interactive change map")
    map_command.add_argument("--before", required=True)
    map_command.add_argument("--after", required=True)
    map_command.add_argument("--clearing", required=True)
    map_command.add_argument("--cleared-hectares", type=float, required=True)
    map_command.add_argument("--config", default="configs/rondonia_example.yaml")
    map_command.add_argument("--output", default="outputs/deforestation_map.html")
    map_command.set_defaults(handler=run_map)

    workflow = subparsers.add_parser("workflow", help="Run the complete local workflow")
    workflow.add_argument("--before-imagery", required=True)
    workflow.add_argument("--before-labels", required=True)
    workflow.add_argument("--after-imagery", required=True)
    workflow.add_argument("--after-labels", required=True)
    workflow.add_argument("--config", default="configs/rondonia_example.yaml")
    workflow.add_argument("--output", default="outputs/workflow")
    workflow.add_argument("--device", default="auto")
    workflow.set_defaults(handler=run_workflow)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

