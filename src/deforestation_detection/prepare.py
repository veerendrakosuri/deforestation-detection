"""Convert aligned imagery and labels into geographically split training tiles."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .raster import assert_aligned, read_raster
from .tiling import extract_tile, geographic_split, iter_tile_windows


@dataclass(frozen=True)
class DatasetSummary:
    train_tiles: int
    validation_tiles: int
    test_tiles: int
    skipped_tiles: int
    band_mean: tuple[float, ...]
    band_std: tuple[float, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def prepare_dataset(
    imagery_path: str | Path,
    label_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
) -> DatasetSummary:
    """Save compressed tiles, a manifest, and train-only normalization stats."""

    imagery = read_raster(imagery_path)
    labels = read_raster(label_path, indexes=(1,))
    assert_aligned(imagery, labels)
    if imagery.array.shape[0] != len(config.imagery.bands):
        raise ValueError(
            f"expected {len(config.imagery.bands)} imagery bands, "
            f"found {imagery.array.shape[0]}"
        )

    height, width = imagery.array.shape[-2:]
    windows = list(
        iter_tile_windows(
            imagery.array.shape,
            tile_size=config.tiling.tile_size,
            stride=config.tiling.stride,
            drop_partial=True,
        )
    )
    splits = geographic_split(
        windows,
        image_height=height,
        image_width=width,
        test_fraction=config.tiling.test_fraction,
        validation_fraction=config.tiling.validation_fraction,
        axis=config.tiling.split_axis,
    )
    split_by_window = {
        window: split_name for split_name, split_windows in splits.items() for window in split_windows
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combined_valid = imagery.valid_mask & labels.valid_mask
    forest = labels.array[0] >= 0.5
    rows: list[dict[str, str | int | float]] = []
    skipped = 0
    band_sum = np.zeros(imagery.array.shape[0], dtype=np.float64)
    band_sum_squares = np.zeros(imagery.array.shape[0], dtype=np.float64)
    band_count = np.zeros(imagery.array.shape[0], dtype=np.int64)

    for window in windows:
        split = split_by_window[window]
        tile_image = extract_tile(imagery.array, window)
        tile_mask = extract_tile(forest, window)
        tile_valid = extract_tile(combined_valid, window)
        valid_fraction = float(np.mean(tile_valid))
        if valid_fraction < config.tiling.minimum_valid_fraction:
            skipped += 1
            continue

        relative_path = Path(split) / f"tile_r{window.row:06d}_c{window.col:06d}.npz"
        tile_path = destination / relative_path
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tile_path,
            image=tile_image.astype(np.float32),
            mask=tile_mask.astype(np.uint8),
            valid=tile_valid.astype(np.uint8),
            row=np.int32(window.row),
            col=np.int32(window.col),
        )

        valid_values = tile_image[:, tile_valid]
        if split == "train" and valid_values.size:
            band_sum += valid_values.sum(axis=1, dtype=np.float64)
            band_sum_squares += np.square(valid_values, dtype=np.float64).sum(axis=1)
            band_count += valid_values.shape[1]

        rows.append(
            {
                "path": relative_path.as_posix(),
                "split": split,
                "row": window.row,
                "col": window.col,
                "valid_fraction": valid_fraction,
                "forest_fraction": float(np.mean(tile_mask[tile_valid])),
            }
        )

    if not np.all(band_count > 0):
        raise ValueError("the training split does not contain valid pixels for every band")
    mean = band_sum / band_count
    variance = np.maximum(band_sum_squares / band_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)

    manifest_path = destination / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "split", "row", "col", "valid_fraction", "forest_fraction"),
        )
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {
        name: sum(row["split"] == name for row in rows)
        for name in ("train", "validation", "test")
    }
    summary = DatasetSummary(
        train_tiles=split_counts["train"],
        validation_tiles=split_counts["validation"],
        test_tiles=split_counts["test"],
        skipped_tiles=skipped,
        band_mean=tuple(float(value) for value in mean),
        band_std=tuple(float(value) for value in std),
    )
    (destination / "dataset.json").write_text(
        json.dumps(
            {
                **summary.to_dict(),
                "bands": list(config.imagery.bands),
                "tile_size": config.tiling.tile_size,
                "stride": config.tiling.stride,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary

