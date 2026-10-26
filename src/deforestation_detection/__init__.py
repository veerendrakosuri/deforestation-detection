"""Deforestation detection from multi-date satellite imagery."""

from .baseline import compute_ndvi, forest_mask_from_ndvi
from .change import ClearingStats, detect_clearing, remove_small_patches
from .metrics import BinaryMetrics, binary_segmentation_metrics

__all__ = [
    "BinaryMetrics",
    "ClearingStats",
    "binary_segmentation_metrics",
    "compute_ndvi",
    "detect_clearing",
    "forest_mask_from_ndvi",
    "remove_small_patches",
]

__version__ = "0.3.1"

