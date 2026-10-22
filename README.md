# Deforestation Detection from Satellite Images

An end-to-end machine-learning project that maps forest cover from Sentinel-2 imagery, compares two dates, and highlights newly cleared areas on an interactive map.

## Project goal

Build and evaluate a U-Net forest-segmentation model, compare it with an NDVI threshold baseline, and estimate the area of newly cleared forest.

## Planned stack

- Python
- Google Earth Engine
- Sentinel-2 Level-2A imagery and Dynamic World forest labels
- NumPy, rasterio, SciPy, and pandas
- PyTorch and segmentation-models-pytorch
- folium/geemap for interactive maps

## Repository layout

```text
data/                 Local imagery and derived datasets (not tracked by Git)
  raw/                Earth Engine exports and source labels
  processed/          Normalized tiles and train/validation/test manifests
models/               Trained checkpoints (not tracked by Git)
notebooks/            Exploration and experiments
outputs/              Metrics, figures, and generated maps
src/
  deforestation_detection/
tests/                Automated tests
```

## Build roadmap

1. Choose one region and two dates from the same season in different years.
2. Create cloud- and shadow-masked Sentinel-2 median composites in Earth Engine.
3. Export six bands: blue, green, red, near-infrared, and two shortwave-infrared bands.
4. Export Dynamic World's `trees` class for the same area and inspect sample labels manually.
5. Build an NDVI baseline and record forest-class IoU and F1.
6. Tile rasters into 256×256 samples and split them by geographic block.
7. Train a U-Net with a ResNet-34 encoder using Dice plus binary cross-entropy loss.
8. Evaluate on a held-out geographic area and inspect failure cases.
9. Compare predictions between dates, remove cleared patches smaller than 50 pixels (about 0.5 ha at 10 m), and calculate hectares cleared.
10. Publish an interactive before/after map with cleared areas highlighted in red.

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

## Important evaluation rules

- Use dates from the same season.
- Mask clouds and cloud shadows.
- Split by geographic area, never by random neighboring tiles.
- Treat Dynamic World labels as model-generated labels and inspect them before training.
- Keep raw data and large model files out of Git.

## Current milestone

The first dependency-light pipeline is implemented:

- NDVI calculation and threshold baseline
- Forest-class IoU, F1, precision, recall, and pixel accuracy
- 256×256 tiling utilities
- Contiguous geographic train/validation/test splits
- Multi-date clearing detection with small-patch filtering
- Cleared-area calculation in hectares
- Synthetic end-to-end demo and automated tests

Run the demo without downloading satellite data:

```powershell
$env:PYTHONPATH = "src"
python -m deforestation_detection demo --output outputs/demo
```

Run the tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The example pilot configuration is in `configs/rondonia_example.yaml`. Its area
and dates are intentionally a starting point and should be inspected in Earth
Engine before a large imagery export.

## Next milestone

1. Authenticate a Google Earth Engine account.
2. Inspect and finalize the pilot area and two same-season date windows.
3. Export cloud-masked Sentinel-2 composites and Dynamic World tree labels.
4. Run the NDVI baseline on the exported rasters.
5. Add the U-Net dataset loader and training loop after validating the labels.

