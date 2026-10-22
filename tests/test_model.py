import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def test_tiny_unet_forward_and_loss(self) -> None:
        import torch

        from deforestation_detection.losses import dice_bce_loss
        from deforestation_detection.model import build_model

        model = build_model(model_name="tiny-unet", in_channels=6, base_channels=4)
        inputs = torch.randn(2, 6, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        valid = torch.ones_like(targets)
        logits = model(inputs)
        loss = dice_bce_loss(logits, targets, valid)

        self.assertEqual(tuple(logits.shape), (2, 1, 32, 32))
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()

