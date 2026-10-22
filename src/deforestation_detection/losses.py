"""Losses for binary forest segmentation."""

from __future__ import annotations


def dice_bce_loss(logits, targets, valid_mask, *, smooth: float = 1.0):
    """Combine masked BCE-with-logits and soft Dice loss."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch is required to calculate training loss") from exc

    valid = valid_mask.to(dtype=logits.dtype)
    valid_count = valid.sum()
    if valid_count.item() == 0:
        raise ValueError("batch contains no valid pixels")

    bce_pixels = functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    bce = (bce_pixels * valid).sum() / valid_count
    probabilities = torch.sigmoid(logits) * valid
    masked_targets = targets * valid
    intersection = (probabilities * masked_targets).sum()
    dice = (2 * intersection + smooth) / (
        probabilities.sum() + masked_targets.sum() + smooth
    )
    return bce + (1 - dice)

