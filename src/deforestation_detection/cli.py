"""Command-line interface for the dependency-light pipeline components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import compute_ndvi, forest_mask_from_ndvi
from .change import detect_clearing
from .metrics import binary_segmentation_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_baseline(args: argparse.Namespace) -> None:
    red = np.load(args.red)
    nir = np.load(args.nir)
    ndvi = compute_ndvi(red, nir)
    forest = forest_mask_from_ndvi(red, nir, threshold=args.threshold)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "ndvi.npy", ndvi)
    np.save(output / "forest_mask.npy", forest)

    summary: dict[str, object] = {
        "threshold": args.threshold,
        "valid_pixels": int(np.count_nonzero(np.isfinite(ndvi))),
        "forest_pixels": int(np.count_nonzero(forest)),
    }
    if args.labels is not None:
        labels = np.load(args.labels)
        valid = np.isfinite(ndvi)
        summary["metrics"] = binary_segmentation_metrics(
            forest, labels, valid_mask=valid
        ).to_dict()

    _write_json(output / "baseline_metrics.json", summary)
    print(json.dumps(summary, indent=2))


def run_change(args: argparse.Namespace) -> None:
    before = np.load(args.before)
    after = np.load(args.after)
    cleared, stats = detect_clearing(
        before,
        after,
        minimum_patch_pixels=args.minimum_patch_pixels,
        pixel_size_m=args.pixel_size_m,
        connectivity=args.connectivity,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "cleared_mask.npy", cleared)
    _write_json(output / "change_metrics.json", stats.to_dict())
    print(json.dumps(stats.to_dict(), indent=2))


def run_demo(args: argparse.Namespace) -> None:
    """Run a small deterministic example without external data."""

    size = 128
    red = np.full((size, size), 0.12, dtype=np.float32)
    nir = np.full((size, size), 0.55, dtype=np.float32)
    labels = np.ones((size, size), dtype=bool)

    red[:24, :24] = 0.25
    nir[:24, :24] = 0.20
    labels[:24, :24] = False

    before = forest_mask_from_ndvi(red, nir, threshold=0.6)
    after = before.copy()
    after[48:68, 54:79] = False
    after[90:93, 90:93] = False

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "demo_red.npy", red)
    np.save(output / "demo_nir.npy", nir)
    np.save(output / "demo_forest_before.npy", before)
    np.save(output / "demo_forest_after.npy", after)

    baseline_metrics = binary_segmentation_metrics(before, labels)
    cleared, change_stats = detect_clearing(
        before,
        after,
        minimum_patch_pixels=50,
        pixel_size_m=10,
    )
    np.save(output / "demo_cleared.npy", cleared)

    summary = {
        "baseline": baseline_metrics.to_dict(),
        "change": change_stats.to_dict(),
    }
    _write_json(output / "demo_metrics.json", summary)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deforestation-detection",
        description="NDVI baseline and forest-loss change detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Create an NDVI forest mask")
    baseline.add_argument("--red", required=True, help="Path to a red-band .npy array")
    baseline.add_argument("--nir", required=True, help="Path to a NIR-band .npy array")
    baseline.add_argument("--labels", help="Optional ground-truth forest-mask .npy array")
    baseline.add_argument("--threshold", type=float, default=0.6)
    baseline.add_argument("--output", default="outputs/baseline")
    baseline.set_defaults(handler=run_baseline)

    change = subparsers.add_parser("change", help="Detect forest loss between two masks")
    change.add_argument("--before", required=True, help="Earlier forest-mask .npy array")
    change.add_argument("--after", required=True, help="Later forest-mask .npy array")
    change.add_argument("--minimum-patch-pixels", type=int, default=50)
    change.add_argument("--pixel-size-m", type=float, default=10.0)
    change.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    change.add_argument("--output", default="outputs/change")
    change.set_defaults(handler=run_change)

    demo = subparsers.add_parser("demo", help="Run the pipeline on synthetic arrays")
    demo.add_argument("--output", default="outputs/demo")
    demo.set_defaults(handler=run_demo)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

