"""Google Earth Engine composite, label, and export helpers."""

from __future__ import annotations

from typing import Any

from .config import DateWindow, PipelineConfig


def _ee():
    try:
        import ee
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("earthengine-api is required for imagery export") from exc
    return ee


def initialize_earth_engine(*, project: str | None, authenticate: bool = False) -> None:
    """Initialize Earth Engine, optionally starting its interactive auth flow."""

    ee = _ee()
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        if not authenticate:
            raise RuntimeError(
                "Earth Engine is not authenticated. Run the export command with "
                "--authenticate from an interactive terminal."
            ) from exc
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def _mask_sentinel2(image: Any, bands: tuple[str, ...]) -> Any:
    """Mask SCL nodata, shadow, cloud, cirrus, and snow classes."""

    scl = image.select("SCL")
    mask = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(mask).select(list(bands)).multiply(0.0001)


def sentinel2_composite(config: PipelineConfig, window: DateWindow) -> Any:
    """Build a cloud-masked median Sentinel-2 surface-reflectance composite."""

    ee = _ee()
    region = ee.Geometry.BBox(*config.region.bbox_wgs84)
    collection = (
        ee.ImageCollection(config.imagery.collection)
        .filterBounds(region)
        .filterDate(window.start, window.end)
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE", config.imagery.cloud_probability_limit
            )
        )
        .map(lambda image: _mask_sentinel2(image, config.imagery.bands))
    )
    return collection.median().resample("bilinear").clip(region)


def dynamic_world_tree_mask(config: PipelineConfig, window: DateWindow) -> Any:
    """Create a binary tree label from median Dynamic World probability."""

    ee = _ee()
    region = ee.Geometry.BBox(*config.region.bbox_wgs84)
    probability = (
        ee.ImageCollection(config.labels.collection)
        .filterBounds(region)
        .filterDate(window.start, window.end)
        .select(config.labels.class_name)
        .median()
        .clip(region)
    )
    return probability.gte(config.labels.tree_probability_threshold).rename("forest").toUint8()


def start_exports(
    config: PipelineConfig,
    *,
    drive_folder: str,
    earth_engine_project: str | None = None,
    authenticate: bool = False,
) -> list[dict[str, str]]:
    """Start aligned before/after imagery and label exports to Google Drive."""

    initialize_earth_engine(project=earth_engine_project, authenticate=authenticate)
    ee = _ee()
    region = ee.Geometry.BBox(*config.region.bbox_wgs84)
    tasks: list[dict[str, str]] = []

    for period_name, window in (
        ("before", config.imagery.before),
        ("after", config.imagery.after),
    ):
        prefix = f"{config.region.name}_{period_name}"
        imagery = sentinel2_composite(config, window).unmask(-9999, sameFootprint=False)
        labels = dynamic_world_tree_mask(config, window).unmask(255, sameFootprint=False)

        imagery_task = ee.batch.Export.image.toDrive(
            image=imagery,
            description=f"{prefix}_sentinel2",
            folder=drive_folder,
            fileNamePrefix=f"{prefix}_sentinel2",
            region=region,
            scale=config.imagery.pixel_size_m,
            crs=config.imagery.crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            skipEmptyTiles=True,
            formatOptions={"cloudOptimized": True, "noData": -9999},
        )
        label_task = ee.batch.Export.image.toDrive(
            image=labels,
            description=f"{prefix}_forest_label",
            folder=drive_folder,
            fileNamePrefix=f"{prefix}_forest_label",
            region=region,
            scale=config.imagery.pixel_size_m,
            crs=config.imagery.crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
            skipEmptyTiles=True,
            formatOptions={"cloudOptimized": True, "noData": 255},
        )
        for kind, task in (("imagery", imagery_task), ("label", label_task)):
            task.start()
            tasks.append(
                {
                    "period": period_name,
                    "kind": kind,
                    "task_id": str(task.id),
                    "description": str(task.config.get("description", "")),
                }
            )
    return tasks

