"""Binary segmentation metrics for the forest class."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float
    iou: float
    pixel_accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def binary_segmentation_metrics(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
) -> BinaryMetrics:
    """Calculate confusion counts and forest-class metrics.

    An empty prediction against an empty target is treated as a perfect result.
    """

    predicted = np.asarray(prediction, dtype=bool)
    expected = np.asarray(target, dtype=bool)
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target must have the same shape")

    if valid_mask is None:
        valid = np.ones(predicted.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != predicted.shape:
            raise ValueError("valid_mask must have the same shape as prediction")

    predicted = predicted[valid]
    expected = expected[valid]
    if predicted.size == 0:
        raise ValueError("no valid pixels were provided")

    true_positive = int(np.count_nonzero(predicted & expected))
    false_positive = int(np.count_nonzero(predicted & ~expected))
    false_negative = int(np.count_nonzero(~predicted & expected))
    true_negative = int(np.count_nonzero(~predicted & ~expected))

    return binary_metrics_from_counts(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )


def binary_metrics_from_counts(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> BinaryMetrics:
    """Build segmentation metrics from accumulated confusion counts."""

    counts = (true_positive, false_positive, false_negative, true_negative)
    if any(value < 0 for value in counts):
        raise ValueError("confusion counts cannot be negative")
    total = sum(counts)
    if total == 0:
        raise ValueError("at least one evaluated pixel is required")

    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else (1.0 if false_negative == 0 else 0.0)
    )
    recall_denominator = true_positive + false_negative
    recall = true_positive / recall_denominator if recall_denominator else 1.0
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = (2 * true_positive / f1_denominator) if f1_denominator else 1.0
    iou_denominator = true_positive + false_positive + false_negative
    iou = true_positive / iou_denominator if iou_denominator else 1.0
    pixel_accuracy = (true_positive + true_negative) / total

    return BinaryMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        pixel_accuracy=pixel_accuracy,
    )

