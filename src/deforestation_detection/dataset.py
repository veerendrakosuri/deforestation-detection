"""PyTorch dataset for prepared NPZ tiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    Dataset = object


def load_dataset_metadata(dataset_dir: str | Path) -> dict:
    return json.loads((Path(dataset_dir) / "dataset.json").read_text(encoding="utf-8"))


class NPZTileDataset(Dataset):
    """Load normalized channel-first tiles from a prepared dataset."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        augment: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for model training")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")

        self.root = Path(dataset_dir)
        self.augment = augment
        metadata = load_dataset_metadata(self.root)
        self.mean = np.asarray(metadata["band_mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(metadata["band_std"], dtype=np.float32)[:, None, None]
        self.rows: list[dict[str, str]] = []
        with (self.root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == split:
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with np.load(self.root / row["path"]) as tile:
            image = tile["image"].astype(np.float32)
            mask = tile["mask"].astype(np.float32)
            valid = tile["valid"].astype(np.float32)

        image = (image - self.mean) / self.std
        image[:, valid < 0.5] = 0
        if self.augment:
            if np.random.random() < 0.5:
                image = image[..., ::-1]
                mask = mask[..., ::-1]
                valid = valid[..., ::-1]
            if np.random.random() < 0.5:
                image = image[..., ::-1, :]
                mask = mask[..., ::-1, :]
                valid = valid[..., ::-1, :]
            rotations = int(np.random.randint(0, 4))
            if rotations:
                image = np.rot90(image, rotations, axes=(-2, -1))
                mask = np.rot90(mask, rotations, axes=(-2, -1))
                valid = np.rot90(valid, rotations, axes=(-2, -1))

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None, ...])),
            "valid": torch.from_numpy(np.ascontiguousarray(valid[None, ...])),
        }

