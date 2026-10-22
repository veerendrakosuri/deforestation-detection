"""U-Net model builders."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


if nn is not None:

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.layers(inputs)


    class TinyUNet(nn.Module):
        """A compact U-Net used for offline smoke tests and CPU experiments."""

        def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
            super().__init__()
            base = base_channels
            self.encoder1 = DoubleConv(in_channels, base)
            self.encoder2 = DoubleConv(base, base * 2)
            self.encoder3 = DoubleConv(base * 2, base * 4)
            self.bottleneck = DoubleConv(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.decoder3 = DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.decoder2 = DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.decoder1 = DoubleConv(base * 2, base)
            self.head = nn.Conv2d(base, 1, kernel_size=1)

        def forward(self, inputs):
            encoder1 = self.encoder1(inputs)
            encoder2 = self.encoder2(self.pool(encoder1))
            encoder3 = self.encoder3(self.pool(encoder2))
            bottleneck = self.bottleneck(self.pool(encoder3))
            decoder3 = self.decoder3(torch.cat((self.up3(bottleneck), encoder3), dim=1))
            decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), dim=1))
            decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), dim=1))
            return self.head(decoder1)

else:

    class TinyUNet:  # pragma: no cover - dependency error path
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to construct a U-Net")


def build_model(
    *,
    model_name: str,
    in_channels: int,
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    base_channels: int = 32,
):
    """Build either the project U-Net or the compact smoke-test model."""

    if torch is None:
        raise RuntimeError("PyTorch is required to construct a model")
    if model_name == "tiny-unet":
        return TinyUNet(in_channels=in_channels, base_channels=base_channels)
    if model_name == "smp-unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("segmentation-models-pytorch is required for smp-unet") from exc
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    raise ValueError("model_name must be 'smp-unet' or 'tiny-unet'")

