# Forest Change Toolkit

[![CI](https://github.com/veerendrakosuri/deforestation-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/veerendrakosuri/deforestation-detection/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An open-source Python library and command-line tool for reproducible forest-change
screening with Sentinel-2 imagery. It builds cloud-masked composites, evaluates an
NDVI baseline, trains a U-Net, compares forest masks from two periods, calculates
potential clearing area, and produces reviewable GeoTIFF and interactive-map outputs.

The PyPI distribution name is `forest-change-toolkit`; the import package remains
`deforestation_detection`, and the preferred terminal command is `forest-change`.

> [!IMPORTANT]
> This is a screening and research tool. Its predictions identify places that may
> require expert or field review; they are not legal proof of deforestation or of
> illegal activity.

## Why use it?

- Run the complete workflow from a terminal or import individual Python functions.
- Start with a deterministic offline demo—no account, imagery, or GPU required.
- Keep study areas and modeling choices in validated YAML configuration.
- Compare a transparent NDVI baseline with a trained segmentation model.
- Preserve georeferencing through prediction, change detection, and map creation.
- Use geographically separated train, validation, and test blocks.

## Quick start

Clone the repository and install the lightweight core:

```powershell
git clone https://github.com/veerendrakosuri/deforestation-detection.git
cd deforestation-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Check the installation and run the offline demo:

```powershell
forest-change --version
forest-change doctor
forest-change demo --output outputs/demo
```

The demo writes NumPy masks and `demo_metrics.json`. It uses synthetic data and must
not be presented as a real-world deforestation result.

## Install feature groups

The core installation supports NumPy baselines, mask comparison, configuration, and
the offline demo. Install only what you need:

```powershell
# Raster processing, maps, and Earth Engine exports
pip install -e ".[geo]"

# Model training and inference
pip install -e ".[ml]"

# Complete local development installation
pip install -e ".[geo,ml,dev]"
```

Large dependencies such as PyTorch and rasterio are deliberately optional. Run
`forest-change doctor` at any time to see which capabilities are ready.

## Configure a study area

Create and validate an editable configuration:

```powershell
forest-change init --output project.yaml
forest-change validate-config --config project.yaml
```

Before running an export, replace the example bounding box, dates, CRS, and thresholds.
Use comparable seasonal windows and inspect the area in Earth Engine.

## Real-data workflow

Earth Engine requires a Google account, an enabled Cloud project, and Earth Engine
access. Authentication is never stored in this repository.

```powershell
forest-change export-earth-engine `
  --config project.yaml `
  --earth-engine-project YOUR_PROJECT_ID `
  --drive-folder deforestation-detection `
  --authenticate
```

After downloading the four aligned GeoTIFF exports from Google Drive:

```powershell
forest-change workflow `
  --before-imagery data/raw/before_sentinel2.tif `
  --before-labels data/raw/before_forest_label.tif `
  --after-imagery data/raw/after_sentinel2.tif `
  --after-labels data/raw/after_forest_label.tif `
  --config project.yaml `
  --output outputs/my-area
```

The workflow creates baseline metrics, geographically split tiles, a trained
checkpoint, held-out metrics, before/after forest masks, a clearing mask, area
statistics, and an interactive HTML review map.

## CLI commands

| Command | Purpose |
|---|---|
| `init` | Write an editable configuration |
| `validate-config` | Validate and summarize configuration |
| `doctor` | Check installed feature groups |
| `demo` | Run an offline synthetic example |
| `baseline` | Create an NDVI mask from NumPy arrays |
| `change` | Compare two NumPy forest masks |
| `export-earth-engine` | Start aligned imagery and label exports |
| `prepare` | Build geographically separated training tiles |
| `train` | Train and evaluate a U-Net |
| `predict` | Produce a georeferenced forest mask |
| `baseline-geotiff` | Evaluate the NDVI baseline on GeoTIFFs |
| `change-geotiff` | Detect change between georeferenced masks |
| `map` | Produce an interactive change-review map |
| `workflow` | Run the complete local pipeline |

Both `forest-change` and the backwards-compatible `deforestation-detection` command
invoke the same CLI. See [the CLI reference](docs/CLI_REFERENCE.md) for examples.

## Python API

The lightweight numerical functions can be used independently:

```python
import numpy as np
from deforestation_detection import detect_clearing, forest_mask_from_ndvi

red = np.load("red.npy")
nir = np.load("nir.npy")
before = forest_mask_from_ndvi(red, nir, threshold=0.6)

after = np.load("forest_after.npy")
cleared, statistics = detect_clearing(
    before,
    after,
    minimum_patch_pixels=50,
    pixel_size_m=10,
)
print(statistics.cleared_hectares)
```

## Pilot results

The included report documents a Rondônia, Brazil pilot with geographically held-out
evaluation data:

| Model | Forest IoU | Forest F1 |
|---|---:|---:|
| NDVI baseline | 0.9109 | 0.9534 |
| U-Net | 0.9529 | 0.9759 |

The run flagged 2,326.06 hectares across 344 patches. Independent screening found
that 74.7% of the detected area was supported by both Dynamic World change and Hansen
Global Forest Change. These results are specific to this pilot and do not demonstrate
accuracy in other regions or seasons. Read [the validation report](docs/VALIDATION_REPORT.md)
for methodology, thresholds, and limitations.

## Data and model limitations

- Dynamic World labels are model-generated land-cover estimates, not field labels.
- Optical imagery can be affected by cloud, haze, shadow, fire, flooding, and seasonality.
- Forest definitions and required minimum mapping units differ between organizations.
- A model trained in one geography may not generalize to another biome.
- Human review and, where important, field verification remain necessary.

## Development

```powershell
pip install -e ".[dev]"
python -m pytest
ruff check .
python -m build
python -m twine check dist/*
```

Generated imagery, credentials, trained checkpoints, and outputs are excluded from
Git. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and the
[methodology](docs/METHODOLOGY.md) before contributing. Maintainers can follow the
[release guide](docs/RELEASING.md) for GitHub and PyPI releases.

## Citation and license

If this software supports published work, cite the metadata in [CITATION.cff](CITATION.cff)
and cite the original satellite and reference datasets used in the analysis.

The software is released under the [MIT License](LICENSE). Dataset providers retain
their own licenses, terms, and attribution requirements.
