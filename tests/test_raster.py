import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("rasterio"), "rasterio not installed")
class RasterTests(unittest.TestCase):
    def test_single_band_writer_discards_invalid_source_block_sizes(self) -> None:
        import rasterio
        from rasterio.transform import from_origin

        from deforestation_detection.raster import read_raster, write_single_band

        test_output = Path("outputs/test-temporary")
        test_output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_output) as directory:
            destination = Path(directory) / "mask.tif"
            profile = {
                "driver": "GTiff",
                "height": 20,
                "width": 20,
                "count": 6,
                "dtype": "float32",
                "crs": "EPSG:32720",
                "transform": from_origin(500_000, 9_000_000, 10, 10),
                "blockxsize": 20,
                "blockysize": 5,
            }
            write_single_band(
                destination,
                np.ones((20, 20), dtype=np.uint8),
                profile,
                dtype="uint8",
                nodata=255,
            )
            result = read_raster(destination)
            self.assertEqual(result.array.shape, (1, 20, 20))
            self.assertTrue(result.valid_mask.all())
            with rasterio.open(destination) as dataset:
                block_height, block_width = dataset.block_shapes[0]
                self.assertEqual(block_height % 16, 0)
                self.assertEqual(block_width % 16, 0)


if __name__ == "__main__":
    unittest.main()

