"""GeoTIFF input, validation, and output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RasterData:
    array: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    profile: dict[str, Any]
    descriptions: tuple[str | None, ...]


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("rasterio is required for GeoTIFF operations") from exc
    return rasterio


def read_raster(path: str | Path, indexes: Sequence[int] | None = None) -> RasterData:
    """Read a raster into channel-first float32 data plus a shared valid mask."""

    rasterio = _rasterio()
    with rasterio.open(path) as dataset:
        selected = list(indexes) if indexes is not None else list(range(1, dataset.count + 1))
        array = dataset.read(selected, out_dtype="float32")
        valid = np.all(dataset.read_masks(selected) > 0, axis=0)
        valid &= np.all(np.isfinite(array), axis=0)
        profile = dataset.profile.copy()
        descriptions = tuple(dataset.descriptions[index - 1] for index in selected)
    return RasterData(array=array, valid_mask=valid, profile=profile, descriptions=descriptions)


def assert_aligned(reference: RasterData, other: RasterData) -> None:
    """Raise when two rasters do not share their grid and coordinate system."""

    keys = ("width", "height", "crs", "transform")
    mismatches = [key for key in keys if reference.profile.get(key) != other.profile.get(key)]
    if mismatches:
        raise ValueError(f"rasters are not aligned; mismatched: {', '.join(mismatches)}")


def write_single_band(
    path: str | Path,
    array: NDArray,
    reference_profile: dict[str, Any],
    *,
    dtype: str,
    nodata: int | float | None,
    description: str | None = None,
) -> Path:
    """Write a georeferenced, compressed, single-band GeoTIFF."""

    rasterio = _rasterio()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = reference_profile.copy()
    # Strip-based source profiles can carry block dimensions that are invalid
    # for tiled GeoTIFF output. Let GDAL choose valid tile dimensions instead.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(destination, "w", **profile) as dataset:
        dataset.write(np.asarray(array, dtype=dtype), 1)
        if description:
            dataset.set_band_description(1, description)
    return destination

