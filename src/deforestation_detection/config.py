"""Validated project configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from importlib import resources
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
    bbox_wgs84: tuple[float, float, float, float] | None = None
    geojson: dict[str, Any] | None = None
    geojson_path: str | None = None

    def __post_init__(self) -> None:
        if (self.bbox_wgs84 is None) == (self.geojson is None):
            raise ValueError("region must define exactly one of bbox_wgs84 or GeoJSON")
        if self.bbox_wgs84 is None:
            return
        west, south, east, north = self.bbox_wgs84
        if west >= east or south >= north:
            raise ValueError("bbox_wgs84 must be [west, south, east, north]")
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("bbox_wgs84 longitudes must be between -180 and 180")
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("bbox_wgs84 latitudes must be between -90 and 90")


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


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section '{key}' must be a mapping")
    return value


def _date_window(mapping: dict[str, Any]) -> DateWindow:
    if not isinstance(mapping, dict):
        raise ValueError("date window must be a mapping with start and end values")
    return DateWindow(start=str(_require(mapping, "start")), end=str(_require(mapping, "end")))


def _validate_geojson(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("GeoJSON must be a JSON object")
    object_type = value.get("type")
    if object_type in {"Polygon", "MultiPolygon"}:
        if not value.get("coordinates"):
            raise ValueError("GeoJSON geometry must contain coordinates")
        return value
    if object_type == "Feature":
        geometry = value.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError("GeoJSON Feature must contain a Polygon or MultiPolygon")
        return value
    if object_type == "FeatureCollection":
        features = value.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("GeoJSON FeatureCollection must contain at least one feature")
        for feature in features:
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not isinstance(geometry, dict) or geometry.get("type") not in {
                "Polygon",
                "MultiPolygon",
            }:
                raise ValueError(
                    "every GeoJSON feature must contain a Polygon or MultiPolygon"
                )
        return value
    raise ValueError(
        "GeoJSON must be a Polygon, MultiPolygon, Feature, or FeatureCollection"
    )


def _region_config(
    region: dict[str, Any],
    *,
    base_dir: Path | None,
) -> RegionConfig:
    choices = (
        region.get("bbox_wgs84") is not None,
        region.get("geojson") is not None,
        region.get("geojson_path") is not None,
    )
    if sum(choices) != 1:
        raise ValueError(
            "region must define exactly one of bbox_wgs84, geojson, or geojson_path"
        )

    name = str(_require(region, "name"))
    if region.get("bbox_wgs84") is not None:
        bbox_value = region["bbox_wgs84"]
        if not isinstance(bbox_value, (list, tuple)):
            raise ValueError("bbox_wgs84 must be a list of four coordinates")
        try:
            bbox = tuple(float(value) for value in bbox_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox_wgs84 coordinates must be numbers") from exc
        if len(bbox) != 4:
            raise ValueError("bbox_wgs84 must contain exactly four coordinates")
        return RegionConfig(name=name, bbox_wgs84=bbox)

    if region.get("geojson") is not None:
        return RegionConfig(name=name, geojson=_validate_geojson(region["geojson"]))

    path_value = str(region["geojson_path"]).strip()
    if not path_value:
        raise ValueError("geojson_path must not be empty")
    configured_path = Path(path_value)
    resolved_path = (
        configured_path
        if configured_path.is_absolute() or base_dir is None
        else base_dir / configured_path
    )
    try:
        geojson = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid GeoJSON in {resolved_path}: {exc}") from exc
    return RegionConfig(
        name=name,
        geojson=_validate_geojson(geojson),
        geojson_path=str(resolved_path.resolve()),
    )


def parse_config(
    raw: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> PipelineConfig:
    project = _require_mapping(raw, "project")
    region = _require_mapping(raw, "region")
    imagery = _require_mapping(raw, "imagery")
    labels = _require_mapping(raw, "labels")
    baseline = _require_mapping(raw, "baseline")
    tiling = _require_mapping(raw, "tiling")
    change = _require_mapping(raw, "change_detection")
    training = _require_mapping(raw, "training")

    config = PipelineConfig(
        name=str(_require(project, "name")),
        random_seed=int(project.get("random_seed", 42)),
        region=_region_config(
            region,
            base_dir=None if base_dir is None else Path(base_dir),
        ),
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
    if date.fromisoformat(config.imagery.before.end) > date.fromisoformat(
        config.imagery.after.start
    ):
        raise ValueError("before date window must end on or before the after window")
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
    if not 0 <= config.tiling.test_fraction < 1:
        raise ValueError("test_fraction must be between 0 (inclusive) and 1 (exclusive)")
    if not 0 <= config.tiling.validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 (inclusive) and 1 (exclusive)"
        )
    if config.tiling.test_fraction + config.tiling.validation_fraction >= 1:
        raise ValueError("test_fraction plus validation_fraction must be less than 1")
    if config.tiling.split_axis not in {"x", "y"}:
        raise ValueError("split_axis must be x or y")
    if config.change_detection.minimum_patch_pixels <= 0:
        raise ValueError("minimum_patch_pixels must be positive")
    if config.change_detection.connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    if config.training.model not in {"smp-unet", "tiny-unet"}:
        raise ValueError("training model must be 'smp-unet' or 'tiny-unet'")
    if config.training.batch_size <= 0 or config.training.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")
    if config.training.learning_rate <= 0 or config.training.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if not 0 <= config.training.prediction_threshold <= 1:
        raise ValueError("prediction_threshold must be between 0 and 1")
    if config.training.base_channels <= 0:
        raise ValueError("base_channels must be positive")
    return config


def _bundled_config_text() -> str:
    return (
        resources.files("deforestation_detection")
        .joinpath("resources/rondonia_example.yaml")
        .read_text(encoding="utf-8")
    )


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Read and validate YAML configuration, using the bundled example by default."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in lean installs
        raise RuntimeError("PyYAML is required to read project configuration") from exc

    config_path = None if path is None else Path(path).resolve()
    source = (
        _bundled_config_text()
        if config_path is None
        else config_path.read_text(encoding="utf-8")
    )
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        source_name = "bundled example" if path is None else str(path)
        raise ValueError(f"invalid YAML in {source_name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return parse_config(
        raw,
        base_dir=None if config_path is None else config_path.parent,
    )


def write_example_config(
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the bundled starter configuration to a user-selected path."""

    output = Path(destination)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists; choose another path or pass --force"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_bundled_config_text(), encoding="utf-8")
    return output

