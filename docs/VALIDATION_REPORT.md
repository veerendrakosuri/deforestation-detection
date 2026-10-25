# Full patch validation report

## Result

All **344 detected clearing patches**, covering **2,326.06 hectares**, were
compared with the Dynamic World tree-label change and the independent Hansen
Global Forest Change loss-year layer.

| Screening result | Patches | Patch share | Area (ha) | Area share |
|---|---:|---:|---:|---:|
| Both references | 151 | 43.9% | 1,736.63 | 74.7% |
| Dynamic World only | 95 | 27.6% | 470.25 | 20.2% |
| Hansen only | 31 | 9.0% | 34.63 | 1.5% |
| Limited or mixed support | 67 | 19.5% | 84.55 | 3.6% |

The patch-average Dynamic World overlap is **61.2%** and the patch-average
Hansen support for 2020-2024 is **40.6%**. Large patches are especially well
supported: the 20 largest cover 1,193.00 hectares, and 19 of them meet the
thresholds for both references.

The number of small mixed-support patches should not be interpreted as the same
amount of uncertainty by area. Patches supported by both references account for
nearly three quarters of all detected hectares, while limited or mixed support
accounts for 3.6%.

## Largest-patch investigation

Patch 1 covers **173.94 hectares** and has **90.8%** overlap with Dynamic World
tree-label change. Its initial Hansen support was only **18.7%** because that
comparison used calendar years 2020-2024.

The annual Hansen check explains most of the mismatch:

| Hansen loss year | Fraction of patch |
|---|---:|
| 2019 | 51.8% |
| 2020 | 9.2% |
| 2021 | 3.3% |
| 2022 | 3.3% |
| 2023 | 2.9% |
| 2024 | 0.0% |

The before Sentinel-2 composite is from July-August 2019, while Hansen reports
only a calendar year. Loss marked in 2019 may therefore be later than the
starting composite. Across 2019-2024, Hansen marks 70.5% of the patch, and the
before/after imagery shows a clear forest-to-cleared transition. The low initial
score is best treated as a timing-boundary warning rather than evidence that the
patch is a false positive. Annual data still cannot determine the exact date.

## Method

The validator ranks 8-connected components in the final clearing mask and
measures two forms of support within each patch:

- Dynamic World support is the fraction overlapping filtered change from tree
  to non-tree in the before/after label rasters.
- Hansen support is the mean of the Landsat-derived `lossyear` mask for calendar
  years 2020-2024 within the patch, sampled at 30.92 metres.

A patch is categorized as supported by both references when Dynamic World
overlap is at least 50% and Hansen support is at least 30%. These are transparent
screening thresholds, not calibrated probabilities. Hansen requests are batched
to make the complete 344-patch review reproducible and reliable.

The independent reference is
[Hansen Global Forest Change v1.13](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13),
asset `UMD/hansen/global_forest_change_2025_v1_13`.

## Reproduce the report

After completing the real-data workflow and authenticating Earth Engine:

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_real_run.py `
  --earth-engine-project YOUR_PROJECT_ID `
  --top-n 344 `
  --preview-limit 20 `
  --earth-engine-batch-size 25
```

The command creates:

- `outputs/rondonia/validation/validation_report.html`
- `outputs/rondonia/validation/validation_map.html`
- `outputs/rondonia/validation/top_patches_contact_sheet.png`
- `outputs/rondonia/validation/largest_patches.csv`
- `outputs/rondonia/validation/largest_patches.geojson`
- `outputs/rondonia/validation/validation_summary.json`

Generated imagery and geospatial artifacts remain outside Git.

## Limits

- Dynamic World supplied the training labels, so it is a consistency reference,
  not an independent validation dataset.
- Hansen uses coarser pixels and calendar-year timing. Its definition is
  stand-replacement disturbance and can include harvest or fire; it is not a
  legal determination of deforestation.
- Screening thresholds were chosen for review and have not been calibrated
  against field observations.
- Field observations or higher-resolution independent imagery are required for
  definitive attribution and exact dating.
