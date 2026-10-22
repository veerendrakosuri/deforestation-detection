import unittest

from deforestation_detection.config import parse_config


class ConfigTests(unittest.TestCase):
    def test_parse_complete_config(self) -> None:
        config = parse_config(
            {
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
        )
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


if __name__ == "__main__":
    unittest.main()

