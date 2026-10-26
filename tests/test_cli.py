import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deforestation_detection.cli import main
from deforestation_detection.config import load_config


class CliTests(unittest.TestCase):
    @staticmethod
    def temporary_directory():
        root = Path("outputs/test-temporary")
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_bundled_config_is_available_outside_repository(self) -> None:
        config = load_config()
        self.assertEqual(config.region.name, "rondonia-pilot")
        self.assertEqual(config.imagery.bands[3], "B8")
        self.assertEqual(config, load_config("configs/rondonia_example.yaml"))

    def test_init_and_validate_config(self) -> None:
        with self.temporary_directory() as directory:
            config_path = Path(directory) / "project.yaml"

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["init", "--output", str(config_path)]), 0)
            self.assertTrue(config_path.is_file())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(["validate-config", "--config", str(config_path)]),
                    0,
                )
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["valid"])
            self.assertEqual(result["region"]["name"], "rondonia-pilot")

    def test_demo_runs_without_optional_dependencies(self) -> None:
        with self.temporary_directory() as directory:
            output = Path(directory) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--output", str(output)]), 0)

            result = json.loads((output / "demo_metrics.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["change"]["cleared_hectares"], 5.0)
            cleared = np.load(output / "demo_cleared.npy")
            self.assertEqual(int(cleared.sum()), 500)

    def test_doctor_has_machine_readable_output(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["doctor", "--json"]), 0)
        report = json.loads(stdout.getvalue())
        self.assertIn("core", report["capabilities"])
        self.assertTrue(report["capabilities"]["core"]["available"])

    def test_missing_config_returns_friendly_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = main(["validate-config", "--config", "does-not-exist.yaml"])
        self.assertEqual(result, 2)
        self.assertIn("error:", stderr.getvalue())

    def test_geojson_config_and_report_commands(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                main(["validate-config", "--config", "configs/geojson_example.yaml"]),
                0,
            )
        validation = json.loads(stdout.getvalue())
        self.assertEqual(validation["region"]["type"], "geojson")
        self.assertEqual(validation["region"]["geojson_type"], "Feature")

        with self.temporary_directory() as directory:
            root = Path(directory)
            results = root / "workflow_results.json"
            report = root / "report.md"
            results.write_text(
                json.dumps(
                    {
                        "provenance": {
                            "version": "0.3.0",
                            "generated_at_utc": "2026-10-26T00:00:00+00:00",
                            "config": {
                                "name": "test",
                                "region": {"name": "pilot"},
                            },
                            "inputs_hashed": False,
                        },
                        "training": {"test_iou": 0.9, "test_f1": 0.95},
                        "dataset": {"train_tiles": 10, "validation_tiles": 2, "test_tiles": 2},
                        "change": {"cleared_hectares": 12.5, "patch_count": 3},
                        "map": "map.html",
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "report",
                            "--results",
                            str(results),
                            "--output",
                            str(report),
                        ]
                    ),
                    0,
                )
            text = report.read_text(encoding="utf-8")
            self.assertIn("12.50 ha", text)
            self.assertIn("legal proof of deforestation", text)


if __name__ == "__main__":
    unittest.main()
