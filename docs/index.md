# Forest Change Toolkit

Forest Change Toolkit is an open-source Python library and command-line application
for reproducible forest-change screening with Sentinel-2 imagery.

It can export aligned data through Google Earth Engine, compare a transparent NDVI
baseline with a U-Net, identify possible forest loss, calculate affected area, and
produce reviewable maps and reports.

> Predictions are screening signals. They require expert review and are not legal
> proof of deforestation or illegal activity.

## Start here

```powershell
git clone https://github.com/veerendrakosuri/deforestation-detection.git
cd deforestation-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
forest-change demo --output outputs/demo
```

Read the [real-data runbook](RUNBOOK.md) for the full process, the
[CLI reference](CLI_REFERENCE.md) for every command, and the
[operations guide](OPERATIONS.md) before scheduling repeated runs.

## What one run produces

- baseline and held-out model metrics;
- georeferenced before/after forest masks;
- a filtered potential-clearing mask and area totals;
- an interactive HTML review map;
- a Markdown screening report;
- machine-readable provenance with optional input hashes.

The repository includes a documented Rondônia pilot. Its measurements describe
that specific pilot only and should not be generalized to other regions or seasons.
