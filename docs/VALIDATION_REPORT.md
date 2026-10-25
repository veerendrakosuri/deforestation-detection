# Largest-patch validation report

## Result

The 20 largest predicted clearing patches cover **1,193.00 hectares**, or
**51.3%** of the 2,326.06 hectares detected by the Rondonia pilot. Nineteen
patches meet the screening thresholds for support from both the Dynamic World
label change and the independent Hansen Global Forest Change loss layer. The
largest patch has strong Dynamic World overlap but lower Hansen support and is
kept as a separate review case.

| Screening result | Patches |
|---|---:|
| Both references | 19 |
| Dynamic World only | 1 |
| Hansen only | 0 |
| Limited or mixed support | 0 |

Across these patches, the mean overlap with Dynamic World tree-label change is
**91.8%**, and the mean Hansen loss fraction is **67.4%**.

## Patch results

| Rank | Area (ha) | Centroid (lat, lon) | Dynamic World | Hansen | Screening result |
|---:|---:|---|---:|---:|---|
| 1 | 173.94 | -9.6096, -63.0162 | 90.8% | 18.7% | Dynamic World only |
| 2 | 104.90 | -9.6292, -63.0883 | 90.2% | 39.4% | Both references |
| 3 | 92.54 | -9.6469, -63.0691 | 87.1% | 51.6% | Both references |
| 4 | 90.07 | -9.7714, -62.9450 | 93.4% | 51.8% | Both references |
| 5 | 77.13 | -9.6762, -62.9198 | 90.5% | 87.5% | Both references |
| 6 | 74.26 | -9.7838, -63.0827 | 90.2% | 89.2% | Both references |
| 7 | 57.81 | -9.7240, -62.9036 | 98.2% | 93.1% | Both references |
| 8 | 53.37 | -9.7218, -62.9171 | 93.1% | 72.4% | Both references |
| 9 | 52.41 | -9.7337, -62.9826 | 94.2% | 76.8% | Both references |
| 10 | 49.69 | -9.6319, -62.9508 | 87.4% | 44.1% | Both references |
| 11 | 45.42 | -9.6903, -62.9724 | 93.7% | 67.6% | Both references |
| 12 | 40.38 | -9.6055, -63.0463 | 94.8% | 66.3% | Both references |
| 13 | 40.32 | -9.6413, -63.0518 | 86.7% | 38.2% | Both references |
| 14 | 39.12 | -9.6886, -62.9306 | 97.6% | 100.0% | Both references |
| 15 | 38.07 | -9.7202, -62.9707 | 95.2% | 95.6% | Both references |
| 16 | 36.45 | -9.7148, -63.0132 | 94.9% | 94.2% | Both references |
| 17 | 34.24 | -9.6456, -62.9608 | 85.0% | 55.2% | Both references |
| 18 | 31.72 | -9.7405, -62.9540 | 86.3% | 69.7% | Both references |
| 19 | 30.85 | -9.7355, -62.9244 | 97.3% | 82.5% | Both references |
| 20 | 30.31 | -9.6260, -62.9389 | 90.3% | 54.7% | Both references |

## Method

The validator ranks 8-connected components in the final clearing mask and
measures two forms of support within each patch:

- Dynamic World support is the fraction that overlaps the filtered change from
  tree to non-tree in the before/after label rasters.
- Hansen support is the mean of the Landsat-derived `lossyear` mask for calendar
  years 2020–2024 within the patch, sampled at 30.92 metres.

A patch is categorized as supported by both references when Dynamic World
overlap is at least 50% and Hansen support is at least 30%. These are transparent
screening thresholds, not calibrated probabilities. The Hansen period starts in
2020 because the before composite is from July–August 2019; this avoids counting
all 2019 loss as later than the starting image, but can miss loss occurring after
that composite during the remainder of 2019.

The independent reference is
[Hansen Global Forest Change v1.13](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13),
asset `UMD/hansen/global_forest_change_2025_v1_13`.

## Reproduce the report

After completing the real-data workflow and authenticating Earth Engine:

```powershell
$env:PYTHONPATH = "src"
python scripts/validate_real_run.py `
  --earth-engine-project YOUR_PROJECT_ID `
  --top-n 20
```

The command creates `outputs/rondonia/validation/validation_report.html`, a
contact sheet and one comparison image per patch, plus CSV, GeoJSON, and JSON
files for further review. Generated data and imagery remain outside Git.

## Limits

- This review covers the largest 20 of 344 detected patches, not every patch.
- Dynamic World supplied the training labels, so it is a consistency reference,
  not an independent validation dataset.
- Hansen uses coarser pixels and calendar-year timing. Its definition is
  stand-replacement disturbance and can include harvest or fire; it is not a
  legal determination of deforestation.
- The images support visual screening, but field observations or higher-resolution
  independent imagery are still required for a definitive claim.
