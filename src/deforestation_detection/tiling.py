"""Spatial tiling and deterministic geographic splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TileWindow:
    row: int
    col: int
    height: int
    width: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.width)


def iter_tile_windows(
    image_shape: Sequence[int],
    *,
    tile_size: int = 256,
    stride: int | None = None,
    drop_partial: bool = True,
) -> Iterator[TileWindow]:
    """Yield windows over the final two dimensions of an image."""

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    stride = tile_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")

    height, width = int(image_shape[-2]), int(image_shape[-1])
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            window_height = min(tile_size, height - row)
            window_width = min(tile_size, width - col)
            if drop_partial and (window_height < tile_size or window_width < tile_size):
                continue
            yield TileWindow(row, col, window_height, window_width)


def extract_tile(array: NDArray, window: TileWindow) -> NDArray:
    """Extract one tile from an array whose last axes are height and width."""

    if array.ndim < 2:
        raise ValueError("array must have at least two dimensions")
    return array[..., window.row_slice, window.col_slice]


def geographic_split(
    windows: Sequence[TileWindow],
    *,
    image_height: int,
    image_width: int,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    axis: Literal["x", "y"] = "x",
) -> dict[str, list[TileWindow]]:
    """Split tiles into contiguous train, validation, and test blocks.

    For ``axis='x'``, western tiles train the model and eastern tiles test it.
    For ``axis='y'``, northern tiles train the model and southern tiles test it.
    """

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if test_fraction < 0 or validation_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be less than 1")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    train_boundary = 1.0 - validation_fraction - test_fraction
    validation_boundary = 1.0 - test_fraction
    result: dict[str, list[TileWindow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for window in windows:
        if axis == "x":
            position = (window.col + window.width / 2) / image_width
        else:
            position = (window.row + window.height / 2) / image_height

        if position < train_boundary:
            result["train"].append(window)
        elif position < validation_boundary:
            result["validation"].append(window)
        else:
            result["test"].append(window)

    return result

