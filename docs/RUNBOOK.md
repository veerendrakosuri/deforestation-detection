# Real-data runbook

## 1. Install

Use Python 3.12 and create a project environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Review the pilot configuration

Open `configs/rondonia_example.yaml` and inspect the bounding box in Earth Engine.
Keep both date windows in the same season. Reduce the area for the first run if
the export would be too large.

## 3. Authenticate and export

Create or select a Google Cloud project with Earth Engine enabled, then run:

```powershell
$env:PYTHONPATH = "src"
python -m deforestation_detection export-earth-engine `
  --config configs/rondonia_example.yaml `
  --earth-engine-project YOUR_PROJECT_ID `
  --drive-folder deforestation-detection `
  --authenticate
```

Four export tasks are started: before/after imagery and before/after labels. Wait
for them to complete, download the GeoTIFFs from Google Drive, and place them in
`data/raw/`.

## 4. Run everything locally

```powershell
python -m deforestation_detection workflow `
  --before-imagery data/raw/rondonia-pilot_before_sentinel2.tif `
  --before-labels data/raw/rondonia-pilot_before_forest_label.tif `
  --after-imagery data/raw/rondonia-pilot_after_sentinel2.tif `
  --after-labels data/raw/rondonia-pilot_after_forest_label.tif `
  --config configs/rondonia_example.yaml `
  --output outputs/rondonia
```

The workflow writes the NDVI results, prepared tiles, best checkpoint, test
metrics, two forest predictions, the clearing mask, and an interactive HTML map.

## 5. Validate the largest clearing patches

Compare the largest connected predictions with Dynamic World label change and
the independent Hansen Global Forest Change loss layer:

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_real_run.py `
  --earth-engine-project YOUR_PROJECT_ID `
  --top-n 344 `
  --preview-limit 20 `
  --earth-engine-batch-size 25
```

Open `outputs/rondonia/validation/validation_report.html` to review the summary,
all-patch table, filterable map, contact sheet, and individual before/after
comparisons for the largest patches. Treat the support categories as screening
evidence rather than field verification.

## 6. Quality checks before reporting results

- Visually compare at least 50 label tiles against the Sentinel-2 composite.
- Confirm train, validation, and test blocks are geographically separate.
- Compare the U-Net against the NDVI baseline on the same valid test pixels.
- Review false positives around rivers, shadows, plantations, and clouds.
- Open the change map and inspect every large clearing patch.
- Replace all `TBD` values in the README only with measured real-data results.

