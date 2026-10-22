import unittest

import numpy as np

from deforestation_detection.metrics import binary_segmentation_metrics


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self) -> None:
        prediction = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
        target = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)

        metrics = binary_segmentation_metrics(prediction, target)

        self.assertEqual(metrics.true_positive, 2)
        self.assertEqual(metrics.false_positive, 1)
        self.assertEqual(metrics.false_negative, 1)
        self.assertEqual(metrics.true_negative, 2)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertAlmostEqual(metrics.f1, 2 / 3)
        self.assertAlmostEqual(metrics.iou, 0.5)

    def test_two_empty_masks_are_perfect(self) -> None:
        metrics = binary_segmentation_metrics(
            np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)
        )
        self.assertEqual(metrics.f1, 1.0)
        self.assertEqual(metrics.iou, 1.0)


if __name__ == "__main__":
    unittest.main()

