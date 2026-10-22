import unittest

import numpy as np

from deforestation_detection.change import detect_clearing, remove_small_patches


class ChangeDetectionTests(unittest.TestCase):
    def test_small_patches_are_removed_and_area_is_reported(self) -> None:
        before = np.ones((20, 20), dtype=bool)
        after = before.copy()
        after[2:7, 3:8] = False
        after[15:17, 15:17] = False

        cleared, stats = detect_clearing(
            before,
            after,
            minimum_patch_pixels=10,
            pixel_size_m=10,
        )

        self.assertEqual(int(np.count_nonzero(cleared)), 25)
        self.assertEqual(stats.patch_count, 1)
        self.assertEqual(stats.patch_sizes_pixels, (25,))
        self.assertAlmostEqual(stats.cleared_hectares, 0.25)

    def test_connectivity_changes_diagonal_components(self) -> None:
        mask = np.eye(3, dtype=bool)
        four_connected, _ = remove_small_patches(
            mask, minimum_pixels=2, connectivity=4
        )
        eight_connected, sizes = remove_small_patches(
            mask, minimum_pixels=2, connectivity=8
        )

        self.assertFalse(four_connected.any())
        self.assertEqual(int(np.count_nonzero(eight_connected)), 3)
        self.assertEqual(sizes, (3,))


if __name__ == "__main__":
    unittest.main()

