import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deforestation_detection.config import load_config
from deforestation_detection.earth_engine import sentinel2_composite


class _FakeComposite:
    def __init__(self):
        self.clipped_to = None

    def resample(self, _method):
        raise AssertionError("reduced collection must not be resampled")

    def clip(self, region):
        self.clipped_to = region
        return self


class _FakeCollection:
    def __init__(self, composite):
        self.composite = composite

    def filterBounds(self, _region):
        return self

    def filterDate(self, _start, _end):
        return self

    def filter(self, _condition):
        return self

    def map(self, _callback):
        return self

    def median(self):
        return self.composite


class EarthEngineCompositeTests(unittest.TestCase):
    def test_collection_composite_is_clipped_without_resampling(self):
        config = load_config("configs/rondonia_example.yaml")
        composite = _FakeComposite()
        region = object()
        fake_ee = SimpleNamespace(
            Geometry=SimpleNamespace(BBox=lambda *_args: region),
            Filter=SimpleNamespace(lte=lambda *_args: object()),
            ImageCollection=lambda _name: _FakeCollection(composite),
        )

        with patch("deforestation_detection.earth_engine._ee", return_value=fake_ee):
            result = sentinel2_composite(config, config.imagery.before)

        self.assertIs(result, composite)
        self.assertIs(composite.clipped_to, region)


if __name__ == "__main__":
    unittest.main()
