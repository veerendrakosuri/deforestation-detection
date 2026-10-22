"""Multi-date deforestation change detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ClearingStats:
    patch_count: int
    cleared_pixels: int
    cleared_hectares: float
    patch_sizes_pixels: tuple[int, ...]

    def to_dict(self) -> dict[str, int | float | tuple[int, ...]]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    raise ValueError("connectivity must be 4 or 8")


def remove_small_patches(
    mask: ArrayLike,
    *,
    minimum_pixels: int = 50,
    connectivity: int = 8,
) -> tuple[NDArray[np.bool_], tuple[int, ...]]:
    """Remove connected components smaller than ``minimum_pixels``.

    This NumPy implementation keeps the core pipeline dependency-light. Large
    production rasters should be processed in tiles or with a raster library.
    """

    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")

    offsets = _neighbor_offsets(connectivity)
    visited = np.zeros(source.shape, dtype=bool)
    filtered = np.zeros(source.shape, dtype=bool)
    kept_sizes: list[int] = []
    height, width = source.shape

    for start_row, start_col in np.argwhere(source):
        if visited[start_row, start_col]:
            continue

        stack = [(int(start_row), int(start_col))]
        visited[start_row, start_col] = True
        component: list[tuple[int, int]] = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))
            for row_offset, col_offset in offsets:
                neighbor_row = row + row_offset
                neighbor_col = col + col_offset
                if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                    continue
                if visited[neighbor_row, neighbor_col] or not source[neighbor_row, neighbor_col]:
                    continue
                visited[neighbor_row, neighbor_col] = True
                stack.append((neighbor_row, neighbor_col))

        if len(component) >= minimum_pixels:
            rows, cols = zip(*component)
            filtered[np.asarray(rows), np.asarray(cols)] = True
            kept_sizes.append(len(component))

    return filtered, tuple(sorted(kept_sizes, reverse=True))


def detect_clearing(
    forest_before: ArrayLike,
    forest_after: ArrayLike,
    *,
    minimum_patch_pixels: int = 50,
    pixel_size_m: float = 10.0,
    connectivity: int = 8,
) -> tuple[NDArray[np.bool_], ClearingStats]:
    """Find forest loss and summarize the retained clearing patches."""

    before = np.asarray(forest_before, dtype=bool)
    after = np.asarray(forest_after, dtype=bool)
    if before.shape != after.shape:
        raise ValueError("forest masks must have the same shape")
    if before.ndim != 2:
        raise ValueError("forest masks must be two-dimensional")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")

    candidates = before & ~after
    cleared, patch_sizes = remove_small_patches(
        candidates,
        minimum_pixels=minimum_patch_pixels,
        connectivity=connectivity,
    )
    cleared_pixels = int(np.count_nonzero(cleared))
    hectares_per_pixel = (pixel_size_m * pixel_size_m) / 10_000
    stats = ClearingStats(
        patch_count=len(patch_sizes),
        cleared_pixels=cleared_pixels,
        cleared_hectares=cleared_pixels * hectares_per_pixel,
        patch_sizes_pixels=patch_sizes,
    )
    return cleared, stats

