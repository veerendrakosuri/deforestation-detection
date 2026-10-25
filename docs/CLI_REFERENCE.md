# CLI reference

The preferred command is `forest-change`. The longer
`deforestation-detection` command remains available for backwards compatibility.

## Global options

```text
forest-change --help
forest-change --version
```

Expected user errors are printed without a Python traceback and return a non-zero
exit status, which makes the commands suitable for scripts and scheduled jobs.

## Setup and diagnostics

### `init`

Write the bundled starter configuration to a new path:

```powershell
forest-change init --output project.yaml
```

Existing files are preserved unless `--force` is explicitly supplied.

### `validate-config`

Validate dates, coordinates, thresholds, tiling, and training settings:

```powershell
forest-change validate-config --config project.yaml
```

Omit `--config` to inspect the configuration bundled with the installed package.

### `doctor`

Show which optional capabilities are installed:

```powershell
forest-change doctor
forest-change doctor --json
forest-change doctor --strict
```

`--strict` returns an error status unless core, geospatial, machine-learning, and
Earth Engine packages are all available.

## Lightweight array commands

These commands require only the core installation.

### `demo`

```powershell
forest-change demo --output outputs/demo
```

Creates deterministic synthetic before/after forest masks, a filtered clearing
mask, and JSON metrics.

### `baseline`

```powershell
forest-change baseline `
  --red red.npy `
  --nir nir.npy `
  --labels labels.npy `
  --threshold 0.6 `
  --output outputs/baseline
```

### `change`

```powershell
forest-change change `
  --before forest_before.npy `
  --after forest_after.npy `
  --minimum-patch-pixels 50 `
  --pixel-size-m 10 `
  --connectivity 8 `
  --output outputs/change
```

## Earth Engine export

```powershell
forest-change export-earth-engine `
  --config project.yaml `
  --earth-engine-project YOUR_PROJECT_ID `
  --drive-folder deforestation-detection `
  --authenticate
```

`--authenticate` starts Google's interactive authentication flow only when needed.
Credentials are managed by the Earth Engine client and must never be committed.

## Dataset and model commands

```powershell
forest-change prepare `
  --imagery before_sentinel2.tif `
  --labels before_forest_label.tif `
  --config project.yaml `
  --output data/processed

forest-change train `
  --dataset data/processed `
  --config project.yaml `
  --output models/run `
  --device auto

forest-change predict `
  --imagery after_sentinel2.tif `
  --checkpoint models/run/best_model.pt `
  --output outputs/predictions/after `
  --device auto
```

`train` can override the configured model, encoder, epochs, batch size, learning
rate, and device. Run `forest-change train --help` for every option.

## GeoTIFF analysis and mapping

```powershell
forest-change baseline-geotiff `
  --imagery before_sentinel2.tif `
  --labels before_forest_label.tif `
  --config project.yaml `
  --output outputs/baseline

forest-change change-geotiff `
  --before forest_before.tif `
  --after forest_after.tif `
  --config project.yaml `
  --output outputs/change

forest-change map `
  --before before_sentinel2.tif `
  --after after_sentinel2.tif `
  --clearing outputs/change/cleared_mask.tif `
  --cleared-hectares 12.34 `
  --config project.yaml `
  --output outputs/change_map.html
```

## Complete workflow

```powershell
forest-change workflow `
  --before-imagery data/raw/before_sentinel2.tif `
  --before-labels data/raw/before_forest_label.tif `
  --after-imagery data/raw/after_sentinel2.tif `
  --after-labels data/raw/after_forest_label.tif `
  --config project.yaml `
  --output outputs/my-area `
  --device auto
```

For scheduled use, archive `workflow_results.json` with the configuration and model
checkpoint so that every result can be traced to its inputs and settings.
