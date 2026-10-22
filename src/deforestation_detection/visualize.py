"""Interactive before/after forest-loss map."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .raster import assert_aligned, read_raster


def _true_color(array: np.ndarray, valid: np.ndarray, rgb_indexes: tuple[int, int, int]):
    rgb = array[list(rgb_indexes)].transpose(1, 2, 0)
    output = np.zeros(rgb.shape, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel][valid]
        if values.size == 0:
            continue
        low, high = np.percentile(values, (2, 98))
        if high <= low:
            high = low + 1e-6
        stretched = np.clip((rgb[..., channel] - low) / (high - low), 0, 1)
        output[..., channel] = np.round(stretched * 255).astype(np.uint8)
    output[~valid] = 0
    return output


def create_change_map(
    before_imagery_path: str | Path,
    after_imagery_path: str | Path,
    cleared_mask_path: str | Path,
    output_path: str | Path,
    *,
    band_names: tuple[str, ...],
    cleared_hectares: float,
    max_dimension: int = 2048,
) -> Path:
    """Save a self-contained Folium map with imagery and clearing overlays."""

    try:
        import folium
        import rasterio
        from rasterio.transform import array_bounds
        from rasterio.warp import transform_bounds
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("folium and rasterio are required to create the map") from exc

    before = read_raster(before_imagery_path)
    after = read_raster(after_imagery_path)
    clearing = read_raster(cleared_mask_path, indexes=(1,))
    assert_aligned(before, after)
    assert_aligned(before, clearing)
    try:
        rgb_indexes = (
            band_names.index("B4"),
            band_names.index("B3"),
            band_names.index("B2"),
        )
    except ValueError as exc:
        raise ValueError("band_names must include B2, B3, and B4") from exc

    height, width = before.array.shape[-2:]
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    west, south, east, north = array_bounds(height, width, before.profile["transform"])
    west, south, east, north = transform_bounds(
        before.profile["crs"], "EPSG:4326", west, south, east, north
    )
    bounds = [[south, west], [north, east]]
    center = [(south + north) / 2, (west + east) / 2]

    step = max(1, int(np.ceil(max(height, width) / max_dimension)))
    before_array = before.array[:, ::step, ::step]
    after_array = after.array[:, ::step, ::step]
    before_valid = before.valid_mask[::step, ::step]
    after_valid = after.valid_mask[::step, ::step]
    clearing_array = clearing.array[0, ::step, ::step]
    clearing_valid = clearing.valid_mask[::step, ::step]
    before_rgb = _true_color(before_array, before_valid, rgb_indexes)
    after_rgb = _true_color(after_array, after_valid, rgb_indexes)
    cleared = (clearing_array >= 0.5) & clearing_valid
    cleared_rgba = np.zeros((*cleared.shape, 4), dtype=np.uint8)
    cleared_rgba[cleared] = (255, 0, 0, 210)

    map_object = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(
        before_rgb,
        bounds=bounds,
        name="Before imagery",
        opacity=1.0,
        show=True,
    ).add_to(map_object)
    folium.raster_layers.ImageOverlay(
        after_rgb,
        bounds=bounds,
        name="After imagery",
        opacity=1.0,
        show=False,
    ).add_to(map_object)
    folium.raster_layers.ImageOverlay(
        cleared_rgba,
        bounds=bounds,
        name="Newly cleared forest",
        opacity=0.9,
        show=True,
    ).add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    folium.Marker(
        center,
        tooltip=f"Detected clearing: {cleared_hectares:.2f} ha",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(map_object)
    map_object.fit_bounds(bounds)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(destination)
    return destination

