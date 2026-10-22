"""End-to-end operations on georeferenced imagery and masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baseline import compute_ndvi, forest_mask_from_ndvi
from .change import detect_clearing
from .config import PipelineConfig
from .metrics import binary_segmentation_metrics
from .raster import assert_aligned, read_raster, write_single_band


def run_ndvi_baseline(
    imagery_path: str | Path,
    label_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
) -> dict:
    """Evaluate NDVI against labels and save georeferenced outputs."""

    imagery = read_raster(imagery_path)
    labels = read_raster(label_path, indexes=(1,))
    assert_aligned(imagery, labels)
    try:
        red_index = config.imagery.bands.index("B4")
        nir_index = config.imagery.bands.index("B8")
    except ValueError as exc:
        raise ValueError("configured imagery bands must include B4 and B8") from exc

    valid = imagery.valid_mask & labels.valid_mask
    red = imagery.array[red_index]
    nir = imagery.array[nir_index]
    ndvi = compute_ndvi(red, nir, valid_mask=valid)
    forest = forest_mask_from_ndvi(
        red, nir, threshold=config.ndvi_threshold, valid_mask=valid
    )
    expected = labels.array[0] >= 0.5
    metrics = binary_segmentation_metrics(forest, expected, valid_mask=valid)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ndvi_output = ndvi.copy()
    ndvi_output[~valid] = -9999
    forest_output = forest.astype(np.uint8)
    forest_output[~valid] = 255
    write_single_band(
        destination / "ndvi.tif",
        ndvi_output,
        imagery.profile,
        dtype="float32",
        nodata=-9999,
        description="ndvi",
    )
    write_single_band(
        destination / "forest_mask.tif",
        forest_output,
        imagery.profile,
        dtype="uint8",
        nodata=255,
        description="ndvi_forest_mask",
    )
    result = {
        "threshold": config.ndvi_threshold,
        "valid_pixels": int(np.count_nonzero(valid)),
        "forest_pixels": int(np.count_nonzero(forest & valid)),
        "metrics": metrics.to_dict(),
    }
    (destination / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_change_detection(
    before_mask_path: str | Path,
    after_mask_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
) -> dict:
    """Detect, filter, georeference, and summarize forest loss."""

    before = read_raster(before_mask_path, indexes=(1,))
    after = read_raster(after_mask_path, indexes=(1,))
    assert_aligned(before, after)
    valid = before.valid_mask & after.valid_mask
    before_forest = (before.array[0] >= 0.5) & valid
    after_forest = (after.array[0] >= 0.5) & valid
    cleared, stats = detect_clearing(
        before_forest,
        after_forest,
        minimum_patch_pixels=config.change_detection.minimum_patch_pixels,
        pixel_size_m=config.imagery.pixel_size_m,
        connectivity=config.change_detection.connectivity,
    )
    output = cleared.astype(np.uint8)
    output[~valid] = 255
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = write_single_band(
        destination / "cleared_mask.tif",
        output,
        before.profile,
        dtype="uint8",
        nodata=255,
        description="newly_cleared_forest",
    )
    result = {**stats.to_dict(), "mask_path": str(mask_path)}
    (destination / "change_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result

