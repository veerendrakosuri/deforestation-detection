# Deforestation Detection from Satellite Images

An end-to-end machine-learning project that maps forest cover from Sentinel-2 imagery, compares two dates, and highlights newly cleared areas on an interactive map.

## Project goal

Build and evaluate a U-Net forest-segmentation model, compare it with an NDVI threshold baseline, and estimate the area of newly cleared forest.

## Stack

- Python
- Google Earth Engine
- Sentinel-2 Level-2A imagery and Dynamic World forest labels
- NumPy and rasterio
- PyTorch and segmentation-models-pytorch
- folium for interactive maps

## Repository layout

```text
data/                 Local imagery and derived datasets (not tracked by Git)
  raw/                Earth Engine exports and source labels
  processed/          Normalized tiles and train/validation/test manifests
models/               Trained checkpoints (not tracked by Git)
notebooks/            Exploration and experiments
outputs/              Metrics, figures, and generated maps
src/
  deforestation_detection/ Core pipeline, model, training, and CLI
configs/              Real-data and smoke-test configurations
docs/                 Methodology and real-data runbook
scripts/              Complete synthetic integration test
tests/                Automated tests
```

## Implemented workflow

1. Build cloud- and shadow-masked, same-season Sentinel-2 median composites in Earth Engine.
2. Export six imagery bands and Dynamic World tree labels as aligned cloud-optimized GeoTIFFs.
3. Evaluate a thresholded NDVI baseline.
4. Tile rasters and create contiguous geographic train, validation, and test blocks.
5. Calculate normalization statistics from training pixels only.
6. Train a six-channel U-Net using Dice plus BCE-with-logits loss.
7. Select the checkpoint by validation IoU and report held-out test metrics.
8. Run sliding-window prediction on both dates.
9. Remove clearing patches smaller than 50 pixels and calculate cleared hectares.
10. Save an interactive before/after map with clearing highlighted in red.

## Evaluation

Report forest-class IoU and F1 on held-out geographic areas for both the NDVI baseline and U-Net. Also report the hectares of clearing found by multi-date change detection.

| Model | Forest IoU | Forest F1 |
|---|---:|---:|
| NDVI baseline | TBD | TBD |
| U-Net | TBD | TBD |

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Earth Engine requires a Google account with Earth Engine access. Authenticate only when you are ready to download imagery.

## Verify the complete pipeline

The smoke test generates aligned synthetic GeoTIFFs, trains the compact U-Net,
runs inference on two dates, detects clearing, and writes an interactive map:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_smoke_test.py
```

Its outputs are written to `outputs/smoke_test/`. These results verify the
software only; they are deliberately synthetic and must not be presented as
real deforestation findings.

Latest verified synthetic run:

| Check | Result |
|---|---:|
| Held-out U-Net IoU | 0.9611 |
| Held-out U-Net F1 | 0.9801 |
| Known clearing | 27.52 ha |
| Detected clearing | 26.31 ha |

Run the automated tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Run with real data

Follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md). After the Earth Engine exports are
downloaded, the complete local pipeline is one command:

```powershell
python -m deforestation_detection workflow `
  --before-imagery data/raw/rondonia-pilot_before_sentinel2.tif `
  --before-labels data/raw/rondonia-pilot_before_forest_label.tif `
  --after-imagery data/raw/rondonia-pilot_after_sentinel2.tif `
  --after-labels data/raw/rondonia-pilot_after_forest_label.tif `
  --config configs/rondonia_example.yaml `
  --output outputs/rondonia
```

## Important evaluation rules

- Use dates from the same season.
- Mask clouds and cloud shadows.
- Split by geographic area, never by random neighboring tiles.
- Treat Dynamic World labels as model-generated labels and inspect them before training.
- Keep raw data and large model files out of Git.

## Project status

The codebase is complete and passes its local unit and end-to-end smoke tests.
Real-data metrics remain `TBD` because Earth Engine authentication, imagery
export, manual label inspection, and GPU training must be performed against the
selected pilot area before any result can be claimed honestly.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for evaluation details and
limitations. The example pilot configuration is `configs/rondonia_example.yaml`.

