import unittest

from deforestation_detection.config import parse_config


class ConfigTests(unittest.TestCase):
    @staticmethod
    def complete_config() -> dict:
        return {
            "project": {"name": "test", "random_seed": 3},
            "region": {"name": "pilot", "bbox_wgs84": [0, 1, 2, 3]},
            "imagery": {
                "collection": "sentinel",
                "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
                "cloud_probability_limit": 20,
                "pixel_size_m": 10,
                "crs": "EPSG:32720",
                "before": {"start": "2020-01-01", "end": "2020-02-01"},
                "after": {"start": "2021-01-01", "end": "2021-02-01"},
            },
            "labels": {"collection": "dw", "class_name": "trees"},
            "baseline": {"ndvi_threshold": 0.6},
            "tiling": {
                "tile_size": 32,
                "test_fraction": 0.2,
                "validation_fraction": 0.1,
            },
            "change_detection": {"minimum_patch_pixels": 50},
            "training": {},
        }

    def test_parse_complete_config(self) -> None:
        config = parse_config(self.complete_config())
        self.assertEqual(config.imagery.bands[3], "B8")
        self.assertEqual(config.tiling.stride, 32)
        self.assertEqual(config.training.model, "smp-unet")

    def test_invalid_bbox_raises(self) -> None:
        raw = {
            "project": {"name": "test"},
            "region": {"name": "bad", "bbox_wgs84": [2, 1, 0, 3]},
            "imagery": {
                "collection": "s",
                "bands": ["B4", "B8"],
                "cloud_probability_limit": 20,
                "pixel_size_m": 10,
                "crs": "EPSG:32720",
                "before": {"start": "2020-01-01", "end": "2020-02-01"},
                "after": {"start": "2021-01-01", "end": "2021-02-01"},
            },
            "labels": {"collection": "d", "class_name": "trees"},
            "baseline": {"ndvi_threshold": 0.6},
            "tiling": {"tile_size": 32, "test_fraction": 0.2, "validation_fraction": 0.1},
            "change_detection": {"minimum_patch_pixels": 50},
            "training": {},
        }
        with self.assertRaises(ValueError):
            parse_config(raw)

    def test_split_fractions_must_leave_training_data(self) -> None:
        raw = self.complete_config()
        raw["tiling"]["test_fraction"] = 0.7
        raw["tiling"]["validation_fraction"] = 0.3
        with self.assertRaisesRegex(ValueError, "must be less than 1"):
            parse_config(raw)

    def test_world_bounds_are_validated(self) -> None:
        raw = self.complete_config()
        raw["region"]["bbox_wgs84"] = [-200, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "longitudes"):
            parse_config(raw)


if __name__ == "__main__":
    unittest.main()

