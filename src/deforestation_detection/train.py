"""Model training and held-out evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dataset import NPZTileDataset, load_dataset_metadata
from .losses import dice_bce_loss
from .metrics import binary_metrics_from_counts
from .model import build_model


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: str
    best_epoch: int
    validation_iou: float
    validation_f1: float
    test_iou: float
    test_f1: float
    device: str

    def to_dict(self) -> dict:
        return asdict(self)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch is required for training") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device, threshold: float) -> tuple[float, dict]:
    import torch

    model.eval()
    total_loss = 0.0
    batches = 0
    true_positive = false_positive = false_negative = true_negative = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            valid = batch["valid"].to(device)
            logits = model(images)
            total_loss += float(dice_bce_loss(logits, targets, valid).item())
            batches += 1

            predicted = torch.sigmoid(logits) >= threshold
            expected = targets >= 0.5
            valid_pixels = valid >= 0.5
            true_positive += int((predicted & expected & valid_pixels).sum().item())
            false_positive += int((predicted & ~expected & valid_pixels).sum().item())
            false_negative += int((~predicted & expected & valid_pixels).sum().item())
            true_negative += int((~predicted & ~expected & valid_pixels).sum().item())

    if batches == 0:
        raise ValueError("evaluation split is empty")
    metrics = binary_metrics_from_counts(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )
    return total_loss / batches, metrics.to_dict()


def train_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "smp-unet",
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    base_channels: int = 32,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    prediction_threshold: float = 0.5,
    seed: int = 42,
    device_name: str = "auto",
) -> TrainingResult:
    """Train a U-Net, select by validation IoU, and evaluate the test split."""

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch is required for training") from exc

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    set_random_seed(seed)
    dataset_path = Path(dataset_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = load_dataset_metadata(dataset_path)
    in_channels = len(metadata["bands"])

    train_dataset = NPZTileDataset(dataset_path, "train", augment=True)
    validation_dataset = NPZTileDataset(dataset_path, "validation")
    test_dataset = NPZTileDataset(dataset_path, "test")
    if not train_dataset or not validation_dataset or not test_dataset:
        raise ValueError("train, validation, and test splits must each contain at least one tile")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        encoder=encoder,
        encoder_weights=encoder_weights,
        base_channels=base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    checkpoint_path = destination / "best_model.pt"
    best_iou = -1.0
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            valid = batch["valid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_bce_loss(logits, targets, valid)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            train_batches += 1

        validation_loss, validation_metrics = _evaluate(
            model, validation_loader, device, prediction_threshold
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / train_batches,
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if validation_metrics["iou"] > best_iou:
            best_iou = float(validation_metrics["iou"])
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "encoder": encoder,
                    "encoder_weights": encoder_weights,
                    "base_channels": base_channels,
                    "in_channels": in_channels,
                    "bands": metadata["bands"],
                    "band_mean": metadata["band_mean"],
                    "band_std": metadata["band_std"],
                    "prediction_threshold": prediction_threshold,
                    "tile_size": metadata["tile_size"],
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    _, test_metrics = _evaluate(model, test_loader, device, prediction_threshold)
    result = TrainingResult(
        checkpoint=str(checkpoint_path),
        best_epoch=best_epoch,
        validation_iou=float(checkpoint["validation_metrics"]["iou"]),
        validation_f1=float(checkpoint["validation_metrics"]["f1"]),
        test_iou=float(test_metrics["iou"]),
        test_f1=float(test_metrics["f1"]),
        device=str(device),
    )
    (destination / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "results.json").write_text(
        json.dumps({**result.to_dict(), "test_metrics": test_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result

