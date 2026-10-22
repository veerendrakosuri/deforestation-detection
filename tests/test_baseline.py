import unittest

import numpy as np

from deforestation_detection.baseline import compute_ndvi, forest_mask_from_ndvi


class BaselineTests(unittest.TestCase):
    def test_ndvi_and_threshold(self) -> None:
        red = np.array([[0.1, 0.3], [0.0, np.nan]], dtype=np.float32)
        nir = np.array([[0.7, 0.4], [0.0, 0.8]], dtype=np.float32)

        ndvi = compute_ndvi(red, nir)
        forest = forest_mask_from_ndvi(red, nir, threshold=0.6)

        self.assertAlmostEqual(float(ndvi[0, 0]), 0.75, places=6)
        self.assertAlmostEqual(float(ndvi[0, 1]), 1 / 7, places=6)
        self.assertTrue(np.isnan(ndvi[1, 0]))
        self.assertTrue(np.isnan(ndvi[1, 1]))
        np.testing.assert_array_equal(
            forest, np.array([[True, False], [False, False]])
        )

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_ndvi(np.zeros((2, 2)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()

