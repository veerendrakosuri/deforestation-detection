"""Sliding-window U-Net inference for GeoTIFF composites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .model import build_model
from .raster import read_raster, write_single_band
from .tiling import iter_tile_windows


@dataclass(frozen=True)
class PredictionResult:
    probability_path: str
    mask_path: str
    valid_pixels: int
    forest_pixels: int

    def to_dict(self) -> dict:
        return asdict(self)


def predict_geotiff(
    imagery_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> PredictionResult:
    """Predict a seamless forest probability raster from a trained checkpoint."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PyTorch is required for inference") from exc

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        ("cpu" if device_name == "auto" else device_name)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        model_name=checkpoint["model_name"],
        in_channels=int(checkpoint["in_channels"]),
        encoder=checkpoint["encoder"],
        encoder_weights=None,
        base_channels=int(checkpoint.get("base_channels", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    raster = read_raster(imagery_path)
    if raster.array.shape[0] != checkpoint["in_channels"]:
        raise ValueError("imagery band count does not match the trained model")
    mean = np.asarray(checkpoint["band_mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(checkpoint["band_std"], dtype=np.float32)[:, None, None]
    tile_size = int(checkpoint["tile_size"])
    threshold = float(checkpoint["prediction_threshold"])
    height, width = raster.array.shape[-2:]
    probability_sum = np.zeros((height, width), dtype=np.float32)
    prediction_count = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for window in iter_tile_windows(
            raster.array.shape,
            tile_size=tile_size,
            stride=tile_size,
            drop_partial=False,
        ):
            tile = raster.array[..., window.row_slice, window.col_slice]
            normalized = (tile - mean) / std
            tile_valid = raster.valid_mask[window.row_slice, window.col_slice]
            normalized[:, ~tile_valid] = 0
            padded = np.zeros(
                (raster.array.shape[0], tile_size, tile_size), dtype=np.float32
            )
            padded[:, : window.height, : window.width] = normalized
            tensor = torch.from_numpy(padded[None, ...]).to(device)
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            probability = probability[: window.height, : window.width]
            probability_sum[window.row_slice, window.col_slice] += probability
            prediction_count[window.row_slice, window.col_slice] += 1

    probability = np.divide(
        probability_sum,
        prediction_count,
        out=np.zeros_like(probability_sum),
        where=prediction_count > 0,
    )
    forest = probability >= threshold
    valid = raster.valid_mask
    probability_output = probability.copy()
    probability_output[~valid] = -9999
    mask_output = forest.astype(np.uint8)
    mask_output[~valid] = 255

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    probability_path = write_single_band(
        destination / "forest_probability.tif",
        probability_output,
        raster.profile,
        dtype="float32",
        nodata=-9999,
        description="forest_probability",
    )
    mask_path = write_single_band(
        destination / "forest_mask.tif",
        mask_output,
        raster.profile,
        dtype="uint8",
        nodata=255,
        description="forest_mask",
    )
    return PredictionResult(
        probability_path=str(probability_path),
        mask_path=str(mask_path),
        valid_pixels=int(np.count_nonzero(valid)),
        forest_pixels=int(np.count_nonzero(forest & valid)),
    )

