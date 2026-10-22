# Methodology

## Objective

Estimate forest cover from Sentinel-2 surface reflectance, evaluate it on a
geographically held-out area, and identify pixels that changed from forest to
non-forest between two same-season composites.

## Data

- Sentinel-2 Level-2A harmonized surface reflectance at 10 m output resolution.
- Six input bands: B2, B3, B4, B8, B11, and B12.
- Dynamic World `trees` probability converted to a binary label at 0.5.
- Median composites over a multi-week window to reduce residual cloud noise.
- Scene Classification Layer values for nodata, defective pixels, cloud shadow,
  medium/high-probability cloud, cirrus, and snow are masked before compositing.

Dynamic World is model-generated supervision, not perfect ground truth. A sample
must be checked visually before any real result is reported.

## Leakage prevention

Tiles are assigned to contiguous geographic blocks. The default split trains on
the western portion and tests on the eastern portion. Neighboring tiles are never
randomly distributed between splits.

Band means and standard deviations are computed from valid training pixels only.
Validation and test imagery do not influence normalization.

## Baseline

The NDVI baseline is:

```text
NDVI = (B8 - B4) / (B8 + B4)
forest = NDVI >= threshold
```

The default real-data threshold is 0.6. It must be evaluated, not tuned on the
test block.

## U-Net

The production configuration uses `segmentation-models-pytorch` U-Net with a
ResNet-34 encoder adapted to six channels. Training uses light flips and 90-degree
rotations plus Dice and binary cross-entropy-with-logits loss. The best checkpoint
is selected by validation forest-class IoU.

The compact in-repository U-Net exists for CPU smoke tests. It is not intended to
replace the production model in reported results.

## Evaluation

Report the following on the untouched geographic test block:

- Forest-class IoU
- Forest-class F1
- Precision and recall
- Pixel accuracy as a secondary metric
- The same IoU and F1 for the NDVI baseline

Inspect the worst tiles for cloud shadow, rivers, plantations, label errors, and
seasonal differences.

## Change detection

A pixel is a clearing candidate when it is predicted as forest before and not
forest after. Eight-connected patches smaller than 50 pixels are removed. At
10 m resolution, every pixel is 0.01 ha, so the default minimum patch is 0.5 ha.

This output is a screening layer, not legal evidence of deforestation. It can be
affected by clouds, fire, harvest cycles, floods, label noise, and model drift.

