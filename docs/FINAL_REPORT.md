# Deforestation Detection from Satellite Images

## Final project report

### Executive summary

This project implements and verifies an end-to-end machine-learning workflow for
detecting forest clearing from multi-date satellite imagery. It uses matched
July-August Sentinel-2 composites from 2019 and 2024 over a pilot area in
Rondonia, Brazil, Dynamic World tree probabilities for training labels, a U-Net
forest-segmentation model, and a thresholded NDVI baseline.

On a geographically held-out test strip, the U-Net achieved forest IoU **0.9529**
and F1 **0.9759**, improving on the NDVI baseline at IoU **0.9109** and F1
**0.9534**. Multi-date inference detected **2,326.06 hectares** of potential
clearing across **344 patches**.

Every patch was screened against two reference signals. **151 patches covering
1,736.63 hectares** were supported by both Dynamic World change and the
independent Hansen Global Forest Change layer. These patches represent **74.7%**
of the detected area. The result is suitable for prioritizing review, but it is
not a field-verified or legal determination of deforestation.

### 1. Objective

The objective was to build a reproducible workflow that:

1. obtains cloud-masked, same-season satellite composites;
2. trains a forest segmentation model using geographically separated data;
3. compares the model against a simple NDVI baseline;
4. detects forest-to-non-forest change between two dates;
5. estimates cleared area and creates an interactive map; and
6. validates every detected patch against additional reference evidence.

### 2. Study area and data

The pilot bounding box is `[-63.10, -9.80, -62.90, -9.60]` in Rondonia,
Brazil. The before period is 1 July to 31 August 2019 and the after period is the
same seasonal window in 2024.

Six Sentinel-2 surface-reflectance bands were used: blue, green, red, near
infrared, short-wave infrared 1, and short-wave infrared 2. Dynamic World tree
probabilities at a threshold of 0.5 supplied forest labels. Cloud and cloud
shadow pixels were removed before median compositing.

The prepared dataset contains 64 non-overlapping 256 by 256-pixel tiles:

| Split | Tiles |
|---|---:|
| Training | 48 |
| Validation | 8 |
| Test | 8 |

The splits are contiguous geographic blocks along the x-axis. Normalization
statistics were computed from training pixels only.

### 3. Methods

The baseline classifies pixels as forest when NDVI is at least 0.6. The learned
model is a six-channel U-Net with a ResNet-34 encoder and ImageNet initialization.
It was trained for 30 CPU epochs using combined Dice and binary cross-entropy
loss. The selected checkpoint is the epoch with highest validation IoU.

For each date, overlapping-window inference creates a forest probability raster
and a thresholded forest mask. Candidate clearing is forest in 2019 and
non-forest in 2024. Eight-connected components smaller than 50 pixels are
removed, equivalent to a minimum retained area of 0.5 hectares at 10-metre
resolution.

### 4. Segmentation evaluation

Both methods were evaluated on the same geographically held-out test pixels.

| Model | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| NDVI baseline | 0.9460 | 0.9608 | 0.9534 | 0.9109 |
| U-Net | 0.9670 | 0.9850 | 0.9759 | 0.9529 |

The U-Net improves IoU by 0.0420 and F1 by 0.0225 over the threshold baseline.
This comparison is meaningful because the test geography and valid-pixel mask
are identical for both methods.

### 5. Change-detection result

The U-Net change mask contains **232,606 pixels**, representing **2,326.06
hectares** in **344 retained patches**. The corresponding Dynamic World label
change contains 2,319.68 hectares in 338 patches.

Predicted change versus label-derived change has precision **0.8523**, recall
**0.8546**, F1 **0.8534**, and IoU **0.7444**. This agreement is a consistency
check because Dynamic World also supplied the training labels.

### 6. Full patch validation

All 344 patches were compared with Dynamic World label change and Hansen loss
years 2020-2024.

| Screening result | Patches | Area (ha) | Area share |
|---|---:|---:|---:|
| Both references | 151 | 1,736.63 | 74.7% |
| Dynamic World only | 95 | 470.25 | 20.2% |
| Hansen only | 31 | 34.63 | 1.5% |
| Limited or mixed | 67 | 84.55 | 3.6% |

The 20 largest patches account for 1,193.00 hectares, or 51.3% of all detected
area. Nineteen of those 20 are supported by both references.

The largest patch illustrates the importance of temporal interpretation. It has
90.8% Dynamic World overlap but only 18.7% Hansen support for 2020-2024. Hansen
assigns another 51.8% of the patch to calendar year 2019. Because the starting
composite is from July-August 2019, annual Hansen data cannot determine whether
that loss occurred before or after the exact starting image. The visual change
and combined 2019-2024 support indicate a timing-boundary issue rather than a
clear false positive.

### 7. Deliverables

The project provides:

- reusable Earth Engine export and local workflow commands;
- trained-model and baseline evaluation metrics;
- GeoTIFF probability, forest, and clearing masks;
- a before/after clearing map;
- a filterable patch-level validation map;
- CSV, GeoJSON, JSON, and image-based validation artifacts;
- automated tests and continuous integration; and
- an online project demonstration.

### 8. Limitations

- Dynamic World labels are generated by another model and are not ground truth.
- Hansen is coarser than Sentinel-2 and records calendar-year stand-replacement
  disturbance, which may include harvest or fire.
- The 50-pixel minimum removes smaller changes and may merge nearby pixels under
  eight-connectivity.
- A single pilot region and seasonal pair do not establish generalization to
  other forest types, years, atmospheric conditions, or sensors.
- Screening categories are deterministic review rules, not probabilities.
- Definitive attribution and exact timing require independent high-resolution
  imagery or field observations.

### 9. Conclusion

The project meets its technical objective: a fully reproducible geospatial ML
pipeline outperforms the NDVI baseline, maps multi-year forest loss, quantifies
area, and validates every retained patch with transparent reference checks.
The strongest result is area-weighted: 74.7% of detected hectares have support
from both reference signals, while limited or mixed support represents only
3.6% of area. The output should be used as a prioritization and screening tool.

### References

- Sentinel-2 Level-2A SR Harmonized:
  <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>
- Dynamic World V1:
  <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1>
- Hansen Global Forest Change v1.13:
  <https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13>
