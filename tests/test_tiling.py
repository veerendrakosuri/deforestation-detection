import unittest

import numpy as np

from deforestation_detection.tiling import (
    extract_tile,
    geographic_split,
    iter_tile_windows,
)


class TilingTests(unittest.TestCase):
    def test_windows_and_extraction(self) -> None:
        array = np.arange(2 * 8 * 8).reshape(2, 8, 8)
        windows = list(iter_tile_windows(array.shape, tile_size=4))

        self.assertEqual(len(windows), 4)
        self.assertEqual(extract_tile(array, windows[0]).shape, (2, 4, 4))
        np.testing.assert_array_equal(
            extract_tile(array, windows[-1]), array[:, 4:8, 4:8]
        )

    def test_partial_tiles_can_be_kept(self) -> None:
        windows = list(
            iter_tile_windows((5, 5), tile_size=4, stride=4, drop_partial=False)
        )
        self.assertEqual(
            [(window.height, window.width) for window in windows],
            [(4, 4), (4, 1), (1, 4), (1, 1)],
        )

    def test_geographic_split_is_contiguous(self) -> None:
        windows = list(iter_tile_windows((10, 100), tile_size=10, stride=10))
        split = geographic_split(
            windows,
            image_height=10,
            image_width=100,
            test_fraction=0.2,
            validation_fraction=0.1,
            axis="x",
        )

        self.assertEqual(len(split["train"]), 7)
        self.assertEqual(len(split["validation"]), 1)
        self.assertEqual(len(split["test"]), 2)
        self.assertTrue(all(window.col < 70 for window in split["train"]))
        self.assertTrue(all(window.col >= 80 for window in split["test"]))


if __name__ == "__main__":
    unittest.main()

