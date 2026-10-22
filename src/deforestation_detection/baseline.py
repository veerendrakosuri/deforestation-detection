"""NDVI-based forest baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_ndvi(
    red: ArrayLike,
    nir: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
    epsilon: float = 1e-6,
) -> NDArray[np.float32]:
    """Compute NDVI while marking invalid pixels as NaN.

    Inputs may be reflectance values or consistently scaled integer bands.
    The arrays must have the same shape.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    red_array = np.asarray(red, dtype=np.float32)
    nir_array = np.asarray(nir, dtype=np.float32)
    if red_array.shape != nir_array.shape:
        raise ValueError("red and nir arrays must have the same shape")

    denominator = nir_array + red_array
    valid = (
        np.isfinite(red_array)
        & np.isfinite(nir_array)
        & (np.abs(denominator) > epsilon)
    )

    if valid_mask is not None:
        mask_array = np.asarray(valid_mask, dtype=bool)
        if mask_array.shape != red_array.shape:
            raise ValueError("valid_mask must have the same shape as the bands")
        valid &= mask_array

    result = np.full(red_array.shape, np.nan, dtype=np.float32)
    np.divide(nir_array - red_array, denominator, out=result, where=valid)
    result[valid] = np.clip(result[valid], -1.0, 1.0)
    return result


def forest_mask_from_ndvi(
    red: ArrayLike,
    nir: ArrayLike,
    *,
    threshold: float = 0.6,
    valid_mask: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Classify valid pixels as forest when NDVI meets ``threshold``."""

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1")

    ndvi = compute_ndvi(red, nir, valid_mask=valid_mask)
    return np.isfinite(ndvi) & (ndvi >= threshold)

