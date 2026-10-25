"""Review the largest predicted clearing patches against independent references."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from deforestation_detection.change import remove_small_patches
from deforestation_detection.config import load_config
from deforestation_detection.raster import assert_aligned, read_raster
from deforestation_detection.visualize import _true_color


HANSEN_DATASET = "UMD/hansen/global_forest_change_2025_v1_13"
HANSEN_CATALOG_URL = (
    "https://developers.google.com/earth-engine/datasets/catalog/"
    "UMD_hansen_global_forest_change_2025_v1_13"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-imagery",
        default="data/raw/rondonia-pilot_before_sentinel2.tif",
    )
    parser.add_argument(
        "--after-imagery",
        default="data/raw/rondonia-pilot_after_sentinel2.tif",
    )
    parser.add_argument(
        "--before-labels",
        default="data/raw/rondonia-pilot_before_forest_label.tif",
    )
    parser.add_argument(
        "--after-labels",
        default="data/raw/rondonia-pilot_after_forest_label.tif",
    )
    parser.add_argument(
        "--clearing",
        default="outputs/rondonia/change/cleared_mask.tif",
    )
    parser.add_argument("--config", default="configs/rondonia_example.yaml")
    parser.add_argument("--output", default="outputs/rondonia/validation")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--earth-engine-project")
    parser.add_argument("--hansen-start-year", type=int, default=2020)
    parser.add_argument("--hansen-end-year", type=int, default=2024)
    return parser.parse_args()


def _merge_geometries(geometries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(geometries) == 1:
        return geometries[0]
    coordinates = []
    for geometry in geometries:
        if geometry["type"] == "Polygon":
            coordinates.append(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            coordinates.extend(geometry["coordinates"])
        else:  # pragma: no cover - rasterio emits polygonal shapes here
            raise ValueError(f"unsupported component geometry: {geometry['type']}")
    return {"type": "MultiPolygon", "coordinates": coordinates}


def _component_geometry(
    component_labels: np.ndarray,
    component_id: int,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    transform,
    crs,
) -> dict[str, Any]:
    import rasterio.features
    import rasterio.windows
    from rasterio.warp import transform_geom

    window = rasterio.windows.Window(
        col_min,
        row_min,
        col_max - col_min,
        row_max - row_min,
    )
    component = (
        component_labels[row_min:row_max, col_min:col_max] == component_id
    )
    window_transform = rasterio.windows.transform(window, transform)
    geometries = [
        geometry
        for geometry, value in rasterio.features.shapes(
            component.astype(np.uint8),
            mask=component,
            transform=window_transform,
            connectivity=8,
        )
        if value == 1
    ]
    projected = _merge_geometries(geometries)
    return transform_geom(crs, "EPSG:4326", projected, precision=6)


def _component_centroid(
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
    crs,
) -> tuple[float, float]:
    from rasterio.warp import transform as transform_coordinates

    row = float(np.mean(rows)) + 0.5
    col = float(np.mean(cols)) + 0.5
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    longitudes, latitudes = transform_coordinates(crs, "EPSG:4326", [x], [y])
    return float(latitudes[0]), float(longitudes[0])


def _extract_patches(
    clearing: np.ndarray,
    label_change: np.ndarray,
    *,
    transform,
    crs,
    pixel_size_m: float,
    top_n: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(2, 2)
    component_labels, component_count = ndimage.label(clearing, structure=structure)
    sizes = np.bincount(component_labels.ravel())
    ranked_ids = sorted(
        range(1, component_count + 1),
        key=lambda component_id: int(sizes[component_id]),
        reverse=True,
    )[:top_n]
    hectares_per_pixel = pixel_size_m * pixel_size_m / 10_000
    patches: list[dict[str, Any]] = []

    for rank, component_id in enumerate(ranked_ids, start=1):
        rows, cols = np.where(component_labels == component_id)
        row_min, row_max = int(rows.min()), int(rows.max()) + 1
        col_min, col_max = int(cols.min()), int(cols.max()) + 1
        pixels = int(rows.size)
        label_pixels = int(np.count_nonzero(label_change[rows, cols]))
        latitude, longitude = _component_centroid(
            rows,
            cols,
            transform,
            crs,
        )
        geometry = _component_geometry(
            component_labels,
            component_id,
            row_min,
            row_max,
            col_min,
            col_max,
            transform,
            crs,
        )
        patches.append(
            {
                "rank": rank,
                "component_id": component_id,
                "pixels": pixels,
                "area_hectares": pixels * hectares_per_pixel,
                "centroid_latitude": latitude,
                "centroid_longitude": longitude,
                "dynamic_world_overlap_pixels": label_pixels,
                "dynamic_world_overlap_fraction": label_pixels / pixels,
                "hansen_loss_fraction": None,
                "hansen_supported_hectares": None,
                "row_min": row_min,
                "row_max": row_max,
                "col_min": col_min,
                "col_max": col_max,
                "geometry": geometry,
            }
        )
    return component_labels, patches


def _add_hansen_support(
    patches: list[dict[str, Any]],
    *,
    earth_engine_project: str,
    start_year: int,
    end_year: int,
) -> None:
    import ee

    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass

    ee.Initialize(project=earth_engine_project)
    start_code = start_year - 2000
    end_code = end_year - 2000
    if not (1 <= start_code <= end_code <= 25):
        raise ValueError("Hansen validation years must be between 2001 and 2025")

    features = [
        ee.Feature(
            ee.Geometry(patch["geometry"], geodesic=False),
            {"rank": patch["rank"]},
        )
        for patch in patches
    ]
    loss_year = ee.Image(HANSEN_DATASET).select("lossyear")
    loss = (
        loss_year.gte(start_code)
        .And(loss_year.lte(end_code))
        .rename("hansen_loss")
    )
    reduced = loss.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.mean(),
        scale=30.92,
        tileScale=4,
    ).getInfo()
    fractions: dict[int, float] = {}
    for feature in reduced["features"]:
        properties = feature["properties"]
        value = properties.get("hansen_loss", properties.get("mean", 0))
        fractions[int(properties["rank"])] = float(value or 0)

    for patch in patches:
        fraction = fractions.get(int(patch["rank"]), 0.0)
        patch["hansen_loss_fraction"] = fraction
        patch["hansen_supported_hectares"] = fraction * patch["area_hectares"]


def _support_category(patch: dict[str, Any]) -> str:
    dynamic_world = float(patch["dynamic_world_overlap_fraction"])
    hansen = float(patch.get("hansen_loss_fraction") or 0)
    if dynamic_world >= 0.5 and hansen >= 0.3:
        return "Both references"
    if dynamic_world >= 0.5:
        return "Dynamic World"
    if hansen >= 0.3:
        return "Hansen"
    return "Limited/mixed"


def _resized_panel(array: np.ndarray, *, size: int = 320):
    from PIL import Image, ImageOps

    image = Image.fromarray(array)
    contained = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), "black")
    offset = ((size - contained.width) // 2, (size - contained.height) // 2)
    panel.paste(contained, offset)
    return panel


def _save_patch_preview(
    patch: dict[str, Any],
    component_labels: np.ndarray,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    label_change: np.ndarray,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    padding = 32
    height, width = component_labels.shape
    row_min = max(0, int(patch["row_min"]) - padding)
    row_max = min(height, int(patch["row_max"]) + padding)
    col_min = max(0, int(patch["col_min"]) - padding)
    col_max = min(width, int(patch["col_max"]) + padding)
    before_crop = before_rgb[row_min:row_max, col_min:col_max]
    after_crop = after_rgb[row_min:row_max, col_min:col_max]
    predicted_crop = (
        component_labels[row_min:row_max, col_min:col_max]
        == int(patch["component_id"])
    )
    label_crop = label_change[row_min:row_max, col_min:col_max]

    predicted_overlay = after_crop.copy().astype(np.float32)
    predicted_overlay[predicted_crop] = (
        0.35 * predicted_overlay[predicted_crop] + 0.65 * np.array([255, 25, 25])
    )
    label_overlay = after_crop.copy().astype(np.float32)
    label_overlay[label_crop] = (
        0.35 * label_overlay[label_crop] + 0.65 * np.array([255, 220, 20])
    )

    titles = ("2019 RGB", "2024 RGB", "Predicted loss", "Dynamic World loss")
    arrays = (
        before_crop,
        after_crop,
        predicted_overlay.astype(np.uint8),
        label_overlay.astype(np.uint8),
    )
    panel_size = 320
    title_height = 25
    canvas = Image.new(
        "RGB",
        (panel_size * len(arrays), panel_size + title_height + 28),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, array) in enumerate(zip(titles, arrays, strict=True)):
        x = index * panel_size
        canvas.paste(_resized_panel(array, size=panel_size), (x, title_height))
        draw.rectangle((x, 0, x + panel_size, title_height), fill="black")
        draw.text((x + 7, 6), title, fill="white")
    caption = (
        f"Patch {patch['rank']:02d} | {patch['area_hectares']:.2f} ha | "
        f"Dynamic World {patch['dynamic_world_overlap_fraction']:.1%} | "
        f"Hansen {float(patch.get('hansen_loss_fraction') or 0):.1%}"
    )
    draw.text((8, panel_size + title_height + 8), caption, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _save_contact_sheet(patch_paths: list[Path], output_path: Path) -> None:
    """Combine the patch panels into one quick-review image."""
    from PIL import Image, ImageOps

    columns = 2
    thumbnail_width = 620
    gap = 12
    thumbnails = []
    for patch_path in patch_paths:
        with Image.open(patch_path) as image:
            converted = image.convert("RGB")
            height = round(converted.height * thumbnail_width / converted.width)
            thumbnails.append(
                converted.resize(
                    (thumbnail_width, height),
                    Image.Resampling.LANCZOS,
                )
            )
    row_height = max(thumbnail.height for thumbnail in thumbnails)
    row_count = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * thumbnail_width + (columns + 1) * gap,
            row_count * row_height + (row_count + 1) * gap,
        ),
        "white",
    )
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        x = gap + column * (thumbnail_width + gap)
        y = gap + row * (row_height + gap)
        sheet.paste(ImageOps.pad(thumbnail, (thumbnail_width, row_height), color="white"), (x, y))
    sheet.save(output_path)


def _public_properties(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": patch["rank"],
        "component_id": patch["component_id"],
        "pixels": patch["pixels"],
        "area_hectares": patch["area_hectares"],
        "centroid_latitude": patch["centroid_latitude"],
        "centroid_longitude": patch["centroid_longitude"],
        "dynamic_world_overlap_pixels": patch["dynamic_world_overlap_pixels"],
        "dynamic_world_overlap_fraction": patch["dynamic_world_overlap_fraction"],
        "hansen_loss_fraction": patch["hansen_loss_fraction"],
        "hansen_supported_hectares": patch["hansen_supported_hectares"],
        "support_category": _support_category(patch),
    }


def _write_outputs(
    patches: list[dict[str, Any]],
    output_dir: Path,
    *,
    hansen_start_year: int,
    hansen_end_year: int,
) -> dict[str, Any]:
    public = [_public_properties(patch) for patch in patches]
    csv_path = output_dir / "largest_patches.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)

    geojson_path = output_dir / "largest_patches.geojson"
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": patch["geometry"],
                "properties": _public_properties(patch),
            }
            for patch in patches
        ],
    }
    geojson_path.write_text(json.dumps(geojson, indent=2) + "\n", encoding="utf-8")

    counts = {
        category: sum(_support_category(patch) == category for patch in patches)
        for category in ("Both references", "Dynamic World", "Hansen", "Limited/mixed")
    }
    summary = {
        "reviewed_patch_count": len(patches),
        "reviewed_area_hectares": sum(float(patch["area_hectares"]) for patch in patches),
        "support_category_counts": counts,
        "mean_dynamic_world_overlap_fraction": float(
            np.mean([patch["dynamic_world_overlap_fraction"] for patch in patches])
        ),
        "mean_hansen_loss_fraction": float(
            np.mean([float(patch.get("hansen_loss_fraction") or 0) for patch in patches])
        ),
        "hansen_dataset": HANSEN_DATASET,
        "hansen_period": [hansen_start_year, hansen_end_year],
        "hansen_catalog_url": HANSEN_CATALOG_URL,
        "artifacts": {
            "csv": str(csv_path),
            "geojson": str(geojson_path),
            "report": str(output_dir / "validation_report.html"),
            "contact_sheet": str(output_dir / "top_patches_contact_sheet.png"),
            "patch_previews": str(output_dir / "patches"),
        },
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = []
    for patch in patches:
        rank = int(patch["rank"])
        latitude = float(patch["centroid_latitude"])
        longitude = float(patch["centroid_longitude"])
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{patch['area_hectares']:.2f}</td>"
            f"<td>{patch['dynamic_world_overlap_fraction']:.1%}</td>"
            f"<td>{float(patch.get('hansen_loss_fraction') or 0):.1%}</td>"
            f"<td>{html.escape(_support_category(patch))}</td>"
            f"<td>{latitude:.6f}, {longitude:.6f}</td>"
            f"<td><a href=\"patches/patch_{rank:03d}.png\">review image</a></td>"
            "</tr>"
        )
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rondonia clearing validation report</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 1180px; margin: 2rem auto; padding: 0 1rem; color: #202124; }}
    h1, h2 {{ color: #174f2b; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
    .card {{ background: #eef7f0; border: 1px solid #bdd8c4; border-radius: 10px; padding: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: .55rem; text-align: left; }}
    th {{ background: #174f2b; color: white; position: sticky; top: 0; }}
    .note {{ background: #fff6dd; border-left: 4px solid #e2a400; padding: 1rem; }}
    code {{ background: #f1f3f4; padding: .1rem .3rem; }}
  </style>
</head>
<body>
  <h1>Rondonia clearing validation report</h1>
  <p>Review of the {len(patches)} largest connected clearing patches from the 2019–2024 pilot.</p>
  <div class="summary">
    <div class="card"><strong>Reviewed area</strong><br>{summary['reviewed_area_hectares']:.2f} ha</div>
    <div class="card"><strong>Mean Dynamic World overlap</strong><br>{summary['mean_dynamic_world_overlap_fraction']:.1%}</div>
    <div class="card"><strong>Mean Hansen support</strong><br>{summary['mean_hansen_loss_fraction']:.1%}</div>
    <div class="card"><strong>Both references</strong><br>{counts['Both references']} patches</div>
  </div>
  <h2>Visual review</h2>
  <p>The contact sheet shows the before image, after image, predicted loss, and Dynamic World loss for every reviewed patch.</p>
  <p><a href="top_patches_contact_sheet.png"><img src="top_patches_contact_sheet.png" alt="Contact sheet of reviewed clearing patches" style="max-width:100%; height:auto;"></a></p>
  <h2>Patch review table</h2>
  <table>
    <thead><tr><th>Rank</th><th>Area (ha)</th><th>Dynamic World</th><th>Hansen</th><th>Support</th><th>Centroid</th><th>Visual</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <h2>Method and limitations</h2>
  <p>Dynamic World overlap compares predicted clearing with the filtered change in its tree labels. Hansen support is the mean of the Landsat-derived <code>lossyear</code> mask for {hansen_start_year}–{hansen_end_year} inside each patch at 30.92 m.</p>
  <p class="note">This is screening evidence, not field validation. Hansen records stand-replacement disturbance, which may include harvest or fire; its resolution and calendar-year timing differ from this Sentinel-2 workflow. Patch images still require human interpretation.</p>
  <p>Independent reference: <a href="{HANSEN_CATALOG_URL}">{HANSEN_DATASET}</a>.</p>
</body>
</html>
"""
    report_path = output_dir / "validation_report.html"
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    if args.top_n <= 0:
        raise ValueError("top-n must be positive")
    config = load_config(args.config)
    before = read_raster(args.before_imagery)
    after = read_raster(args.after_imagery)
    before_labels = read_raster(args.before_labels, indexes=(1,))
    after_labels = read_raster(args.after_labels, indexes=(1,))
    clearing = read_raster(args.clearing, indexes=(1,))
    for raster in (after, before_labels, after_labels, clearing):
        assert_aligned(before, raster)

    valid = before_labels.valid_mask & after_labels.valid_mask
    label_candidates = (
        (before_labels.array[0] >= 0.5)
        & (after_labels.array[0] < 0.5)
        & valid
    )
    label_change, _ = remove_small_patches(
        label_candidates,
        minimum_pixels=config.change_detection.minimum_patch_pixels,
        connectivity=config.change_detection.connectivity,
    )
    predicted_change = (clearing.array[0] >= 0.5) & clearing.valid_mask
    component_labels, patches = _extract_patches(
        predicted_change,
        label_change,
        transform=before.profile["transform"],
        crs=before.profile["crs"],
        pixel_size_m=config.imagery.pixel_size_m,
        top_n=args.top_n,
    )
    if not patches:
        raise ValueError("no clearing patches were found")

    if not args.earth_engine_project:
        raise ValueError("--earth-engine-project is required for Hansen validation")
    _add_hansen_support(
        patches,
        earth_engine_project=args.earth_engine_project,
        start_year=args.hansen_start_year,
        end_year=args.hansen_end_year,
    )

    try:
        rgb_indexes = (
            config.imagery.bands.index("B4"),
            config.imagery.bands.index("B3"),
            config.imagery.bands.index("B2"),
        )
    except ValueError as exc:
        raise ValueError("configured bands must include B2, B3, and B4") from exc
    before_rgb = _true_color(before.array, before.valid_mask, rgb_indexes)
    after_rgb = _true_color(after.array, after.valid_mask, rgb_indexes)
    output_dir = Path(args.output)
    patch_dir = output_dir / "patches"
    patch_paths = []
    for patch in patches:
        patch_path = patch_dir / f"patch_{int(patch['rank']):03d}.png"
        _save_patch_preview(
            patch,
            component_labels,
            before_rgb,
            after_rgb,
            label_change,
            patch_path,
        )
        patch_paths.append(patch_path)
    _save_contact_sheet(patch_paths, output_dir / "top_patches_contact_sheet.png")
    summary = _write_outputs(
        patches,
        output_dir,
        hansen_start_year=args.hansen_start_year,
        hansen_end_year=args.hansen_end_year,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
