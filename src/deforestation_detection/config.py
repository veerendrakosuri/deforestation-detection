"""Validated project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DateWindow:
    start: str
    end: str

    def __post_init__(self) -> None:
        start_date = date.fromisoformat(self.start)
        end_date = date.fromisoformat(self.end)
        if start_date >= end_date:
            raise ValueError("date-window start must be earlier than end")


@dataclass(frozen=True)
class RegionConfig:
    name: str
    bbox_wgs84: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        west, south, east, north = self.bbox_wgs84
        if west >= east or south >= north:
            raise ValueError("bbox_wgs84 must be [west, south, east, north]")


@dataclass(frozen=True)
class ImageryConfig:
    collection: str
    bands: tuple[str, ...]
    cloud_probability_limit: float
    pixel_size_m: float
    crs: str
    before: DateWindow
    after: DateWindow


@dataclass(frozen=True)
class LabelConfig:
    collection: str
    class_name: str
    tree_probability_threshold: float


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int
    stride: int
    test_fraction: float
    validation_fraction: float
    split_axis: str
    minimum_valid_fraction: float


@dataclass(frozen=True)
class ChangeConfig:
    minimum_patch_pixels: int
    connectivity: int


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    encoder: str
    encoder_weights: str | None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    prediction_threshold: float
    base_channels: int


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    random_seed: int
    region: RegionConfig
    imagery: ImageryConfig
    labels: LabelConfig
    ndvi_threshold: float
    tiling: TilingConfig
    change_detection: ChangeConfig
    training: TrainingConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    return mapping[key]


def _date_window(mapping: dict[str, Any]) -> DateWindow:
    return DateWindow(start=str(_require(mapping, "start")), end=str(_require(mapping, "end")))


def parse_config(raw: dict[str, Any]) -> PipelineConfig:
    project = _require(raw, "project")
    region = _require(raw, "region")
    imagery = _require(raw, "imagery")
    labels = _require(raw, "labels")
    baseline = _require(raw, "baseline")
    tiling = _require(raw, "tiling")
    change = _require(raw, "change_detection")
    training = _require(raw, "training")

    bbox = tuple(float(value) for value in _require(region, "bbox_wgs84"))
    if len(bbox) != 4:
        raise ValueError("bbox_wgs84 must contain exactly four coordinates")

    config = PipelineConfig(
        name=str(_require(project, "name")),
        random_seed=int(project.get("random_seed", 42)),
        region=RegionConfig(name=str(_require(region, "name")), bbox_wgs84=bbox),
        imagery=ImageryConfig(
            collection=str(_require(imagery, "collection")),
            bands=tuple(str(band) for band in _require(imagery, "bands")),
            cloud_probability_limit=float(_require(imagery, "cloud_probability_limit")),
            pixel_size_m=float(_require(imagery, "pixel_size_m")),
            crs=str(_require(imagery, "crs")),
            before=_date_window(_require(imagery, "before")),
            after=_date_window(_require(imagery, "after")),
        ),
        labels=LabelConfig(
            collection=str(_require(labels, "collection")),
            class_name=str(_require(labels, "class_name")),
            tree_probability_threshold=float(
                labels.get("tree_probability_threshold", 0.5)
            ),
        ),
        ndvi_threshold=float(_require(baseline, "ndvi_threshold")),
        tiling=TilingConfig(
            tile_size=int(_require(tiling, "tile_size")),
            stride=int(tiling.get("stride", _require(tiling, "tile_size"))),
            test_fraction=float(_require(tiling, "test_fraction")),
            validation_fraction=float(_require(tiling, "validation_fraction")),
            split_axis=str(tiling.get("split_axis", "x")),
            minimum_valid_fraction=float(tiling.get("minimum_valid_fraction", 0.8)),
        ),
        change_detection=ChangeConfig(
            minimum_patch_pixels=int(_require(change, "minimum_patch_pixels")),
            connectivity=int(change.get("connectivity", 8)),
        ),
        training=TrainingConfig(
            model=str(training.get("model", "smp-unet")),
            encoder=str(training.get("encoder", "resnet34")),
            encoder_weights=(
                None
                if training.get("encoder_weights") in {None, "none", "None"}
                else str(training["encoder_weights"])
            ),
            batch_size=int(training.get("batch_size", 8)),
            epochs=int(training.get("epochs", 30)),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            prediction_threshold=float(training.get("prediction_threshold", 0.5)),
            base_channels=int(training.get("base_channels", 32)),
        ),
    )

    if not config.imagery.bands:
        raise ValueError("at least one imagery band is required")
    if not 0 <= config.imagery.cloud_probability_limit <= 100:
        raise ValueError("cloud_probability_limit must be between 0 and 100")
    if config.imagery.pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    if not 0 <= config.labels.tree_probability_threshold <= 1:
        raise ValueError("tree_probability_threshold must be between 0 and 1")
    if not -1 <= config.ndvi_threshold <= 1:
        raise ValueError("ndvi_threshold must be between -1 and 1")
    if config.tiling.tile_size <= 0 or config.tiling.stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if not 0 <= config.tiling.minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be between 0 and 1")
    if config.tiling.split_axis not in {"x", "y"}:
        raise ValueError("split_axis must be x or y")
    if config.change_detection.connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    if config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if config.training.learning_rate <= 0 or config.training.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0 <= config.training.prediction_threshold <= 1:
        raise ValueError("prediction_threshold must be between 0 and 1")
    if config.training.base_channels <= 0:
        raise ValueError("base_channels must be positive")
    return config


def load_config(path: str | Path) -> PipelineConfig:
    """Read and validate a YAML configuration file."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in lean installs
        raise RuntimeError("PyYAML is required to read project configuration") from exc

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return parse_config(raw)

