"""Human-readable screening reports built from workflow result JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _metric(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _training_table(training: dict[str, Any]) -> list[str]:
    return [
        "| Metric | Value |",
        "|---|---:|",
        f"| Best epoch | {_metric(training.get('best_epoch'))} |",
        f"| Validation IoU | {_metric(training.get('validation_iou'))} |",
        f"| Validation F1 | {_metric(training.get('validation_f1'))} |",
        f"| Test IoU | {_metric(training.get('test_iou'))} |",
        f"| Test F1 | {_metric(training.get('test_f1'))} |",
        f"| Device | {_metric(training.get('device'))} |",
    ]


def build_workflow_report(results: dict[str, Any]) -> str:
    """Convert a workflow result mapping into a concise Markdown report."""

    if not isinstance(results, dict):
        raise ValueError("workflow results must be a JSON object")
    training = results.get("training") or {}
    change = results.get("change") or {}
    dataset = results.get("dataset") or {}
    provenance = results.get("provenance") or {}
    config = provenance.get("config") or {}
    region = config.get("region") or {}

    lines = [
        "# Forest-change screening report",
        "",
        "> This report identifies potential forest change for expert review. It is not",
        "> legal proof of deforestation or illegal activity.",
        "",
        "## Run summary",
        "",
        f"- Project: {_metric(config.get('name'))}",
        f"- Region: {_metric(region.get('name'))}",
        f"- Toolkit version: {_metric(provenance.get('version'))}",
        f"- Generated (UTC): {_metric(provenance.get('generated_at_utc'))}",
        f"- Potential clearing: {_metric(change.get('cleared_hectares'), digits=2)} ha",
        f"- Retained patches: {_metric(change.get('patch_count'))}",
        "",
        "## Model evaluation",
        "",
        *_training_table(training),
        "",
        "## Dataset",
        "",
        f"- Training tiles: {_metric(dataset.get('train_tiles'))}",
        f"- Validation tiles: {_metric(dataset.get('validation_tiles'))}",
        f"- Test tiles: {_metric(dataset.get('test_tiles'))}",
        f"- Skipped tiles: {_metric(dataset.get('skipped_tiles'))}",
        "",
        "## Outputs",
        "",
        f"- Interactive map: {_metric(results.get('map'))}",
        f"- Clearing mask: {_metric(change.get('mask_path'))}",
        f"- Model checkpoint: {_metric(training.get('checkpoint'))}",
        "",
        "## Interpretation requirements",
        "",
        "- Review large patches against the original imagery and independent references.",
        "- Check clouds, shadows, fire, flooding, plantations, and seasonal differences.",
        "- Record expert review and field evidence before operational or legal decisions.",
    ]
    if provenance.get("inputs_hashed"):
        lines.extend(
            [
                "",
                "## Input integrity",
                "",
                "SHA-256 hashes are recorded in `workflow_results.json`.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_workflow_report(
    results_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Read workflow JSON and write a Markdown screening report."""

    source = Path(results_path)
    try:
        results = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid workflow JSON in {source}: {exc}") from exc
    destination = (
        source.with_name("screening_report.md")
        if output_path is None
        else Path(output_path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_workflow_report(results), encoding="utf-8")
    return destination
