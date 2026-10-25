"""Build the polished PDF report for the completed Rondonia pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GREEN = "#174F2B"
LIGHT_GREEN = "#EAF5ED"
AMBER = "#D99100"
BLUE = "#2563EB"
RED = "#C73E3A"
INK = "#172026"
MUTED = "#5B6770"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-results",
        default="outputs/rondonia/workflow_results.json",
    )
    parser.add_argument(
        "--validation-summary",
        default="outputs/rondonia/validation/validation_summary.json",
    )
    parser.add_argument(
        "--largest-patch-image",
        default="outputs/rondonia/validation/patches/patch_001.png",
    )
    parser.add_argument(
        "--output",
        default="output/pdf/deforestation_detection_final_report.pdf",
    )
    parser.add_argument("--temporary-directory", default="tmp/pdfs")
    return parser.parse_args()


def _make_charts(workflow: dict, validation: dict, temporary: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw, ImageFont

    temporary.mkdir(parents=True, exist_ok=True)

    def font(size: int, *, bold: bool = False):
        candidates = (
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ),
        )
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    title_font = font(38, bold=True)
    label_font = font(25)
    small_font = font(22)
    value_font = font(22, bold=True)
    axis_color = "#AAB4BA"
    grid_color = "#E1E6E3"

    baseline = workflow["baseline_test"]["metrics"]
    unet = workflow["training"]["test_metrics"]
    labels = ("Precision", "Recall", "F1", "IoU")
    baseline_values = [baseline[key.lower()] for key in labels]
    unet_values = [unet[key.lower()] for key in labels]
    canvas = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 45), "Forest segmentation performance", font=title_font, fill=INK)
    left, top, right, bottom = 125, 150, 1525, 600
    for tick in (0.82, 0.86, 0.90, 0.94, 0.98, 1.0):
        y = bottom - (tick - 0.82) / 0.18 * (bottom - top)
        draw.line((left, y, right, y), fill=grid_color, width=2)
        draw.text((48, y - 13), f"{tick:.2f}", font=small_font, fill=MUTED)
    draw.line((left, top, left, bottom), fill=axis_color, width=3)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=3)
    group_width = (right - left) / len(labels)
    bar_width = 90
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        for shift, value, color in (
            (-bar_width / 2, baseline_values[index], "#A7B1B7"),
            (bar_width / 2, unet_values[index], GREEN),
        ):
            x0 = center + shift - bar_width / 2
            y0 = bottom - (value - 0.82) / 0.18 * (bottom - top)
            draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=color)
            text = f"{value:.3f}"
            text_width = draw.textlength(text, font=value_font)
            draw.text((x0 + (bar_width - text_width) / 2, y0 - 32), text, font=value_font, fill=INK)
        text_width = draw.textlength(label, font=label_font)
        draw.text((center - text_width / 2, bottom + 18), label, font=label_font, fill=MUTED)
    draw.rectangle((1210, 80, 1240, 105), fill="#A7B1B7")
    draw.text((1250, 78), "NDVI", font=small_font, fill=MUTED)
    draw.rectangle((1360, 80, 1390, 105), fill=GREEN)
    draw.text((1400, 78), "U-Net", font=small_font, fill=MUTED)
    model_chart = temporary / "model_metrics.png"
    canvas.save(model_chart)

    order = ("Both references", "Dynamic World", "Hansen", "Limited/mixed")
    area_fractions = validation["support_category_area_fraction"]
    chart_colors = ("#16A34A", AMBER, BLUE, RED)
    values = [100 * area_fractions[key] for key in order]
    canvas = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 45), "Reference support across all 2,326.06 hectares", font=title_font, fill=INK)
    left, top, right, bottom = 430, 155, 1510, 600
    for tick in range(0, 81, 20):
        x = left + tick / 80 * (right - left)
        draw.line((x, top, x, bottom), fill=grid_color, width=2)
        draw.text((x - 14, bottom + 18), str(tick), font=small_font, fill=MUTED)
    bar_height = 68
    gap = 38
    for index, (label, value, color) in enumerate(zip(order, values, chart_colors, strict=True)):
        y = top + index * (bar_height + gap)
        draw.text((70, y + 17), label, font=label_font, fill=INK)
        x1 = left + value / 80 * (right - left)
        draw.rounded_rectangle((left, y, x1, y + bar_height), radius=10, fill=color)
        draw.text((x1 + 16, y + 18), f"{value:.1f}%", font=value_font, fill=INK)
    draw.text((1040, 660), "Share of detected area (%)", font=small_font, fill=MUTED)
    validation_chart = temporary / "validation_area.png"
    canvas.save(validation_chart)

    annual = validation["largest_patch_review"]["hansen_annual_loss_fraction"]
    years = [int(year) for year in annual]
    annual_values = [100 * annual[str(year)] for year in years]
    canvas = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 45), "Largest patch: Hansen loss timing", font=title_font, fill=INK)
    left, top, right, bottom = 125, 150, 1525, 600
    for tick in range(0, 61, 15):
        y = bottom - tick / 60 * (bottom - top)
        draw.line((left, y, right, y), fill=grid_color, width=2)
        draw.text((55, y - 13), f"{tick}%", font=small_font, fill=MUTED)
    draw.line((left, top, left, bottom), fill=axis_color, width=3)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=3)
    group_width = (right - left) / len(years)
    bar_width = 120
    for index, (year, value) in enumerate(zip(years, annual_values, strict=True)):
        center = left + group_width * (index + 0.5)
        x0 = center - bar_width / 2
        y0 = bottom - value / 60 * (bottom - top)
        color = AMBER if year == 2019 else GREEN
        draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=color)
        text = f"{value:.1f}%"
        text_width = draw.textlength(text, font=value_font)
        draw.text((center - text_width / 2, y0 - 32), text, font=value_font, fill=INK)
        year_text = str(year)
        year_width = draw.textlength(year_text, font=label_font)
        draw.text((center - year_width / 2, bottom + 18), year_text, font=label_font, fill=MUTED)
    timing_chart = temporary / "largest_patch_timing.png"
    canvas.save(timing_chart)

    return {
        "model": model_chart,
        "validation": validation_chart,
        "timing": timing_chart,
    }


def _build_pdf(
    workflow: dict,
    validation: dict,
    largest_patch_image: Path,
    charts: dict[str, Path],
    output: Path,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=33,
            textColor=colors.HexColor(GREEN),
            alignment=TA_LEFT,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor(MUTED),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor(GREEN),
            spaceBefore=8,
            spaceAfter=8,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor(INK),
            spaceBefore=7,
            spaceAfter=5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor(INK),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.5,
            textColor=colors.HexColor(MUTED),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricNumber",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=22,
            textColor=colors.HexColor(GREEN),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetricLabel",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(MUTED),
            alignment=TA_CENTER,
        )
    )

    def body(text: str) -> Paragraph:
        return Paragraph(text, styles["BodyReport"])

    def section(text: str) -> Paragraph:
        return Paragraph(text, styles["Section"])

    def bullet(text: str) -> Paragraph:
        return Paragraph(f"- {text}", styles["BodyReport"])

    def data_table(data: list[list[object]], widths: list[float]) -> Table:
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(GREEN)),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                    ("LEADING", (0, 0), (-1, -1), 10),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F4")]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCC7C0")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5CE"))
        canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, 9.5 * mm, "Rondonia deforestation detection - final report")
        canvas.drawRightString(192 * mm, 9.5 * mm, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Deforestation Detection from Satellite Images",
        author="Veerendra Kosuri",
        subject="Final machine-learning project report",
    )

    training = workflow["training"]
    change = workflow["change"]
    story = [
        Spacer(1, 18 * mm),
        Paragraph("DEFORESTATION DETECTION", styles["SmallReport"]),
        Paragraph("From satellite images to validated clearing patches", styles["ReportTitle"]),
        Paragraph(
            "Final project report | Rondonia pilot | Sentinel-2, Dynamic World, U-Net, and Hansen Global Forest Change",
            styles["ReportSubtitle"],
        ),
    ]
    metric_cards = Table(
        [
            [
                Paragraph(f"{training['test_iou']:.4f}", styles["MetricNumber"]),
                Paragraph(f"{change['cleared_hectares']:,.2f}", styles["MetricNumber"]),
                Paragraph(str(change["patch_count"]), styles["MetricNumber"]),
                Paragraph("74.7%", styles["MetricNumber"]),
            ],
            [
                Paragraph("Held-out U-Net IoU", styles["MetricLabel"]),
                Paragraph("Detected hectares", styles["MetricLabel"]),
                Paragraph("Validated patches", styles["MetricLabel"]),
                Paragraph("Area supported by both references", styles["MetricLabel"]),
            ],
        ],
        colWidths=[43.5 * mm] * 4,
        rowHeights=[12 * mm, 12 * mm],
    )
    metric_cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(LIGHT_GREEN)),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9D4C0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFE1D3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend(
        [
            metric_cards,
            Spacer(1, 10 * mm),
            section("Executive summary"),
            body(
                "This project delivers an end-to-end machine-learning workflow for mapping forest cover and screening potential clearing between matched 2019 and 2024 Sentinel-2 composites. The U-Net outperforms the NDVI baseline on a geographically held-out strip, and all 344 retained clearing patches are compared with two reference signals."
            ),
            body(
                "The output is designed for prioritization and review. It is not field verification or a legal determination of deforestation."
            ),
            PageBreak(),
            section("1. Objective and study design"),
            body(
                "The pilot covers a 0.2 by 0.2 degree area in Rondonia, Brazil (bounding box -63.10, -9.80, -62.90, -9.60). Same-season July-August composites reduce seasonal bias between the 2019 before period and the 2024 after period."
            ),
            bullet("Cloud- and shadow-masked Sentinel-2 Level-2A surface reflectance."),
            bullet("Six bands: blue, green, red, near infrared, SWIR1, and SWIR2."),
            bullet("Dynamic World tree probability at 0.5 for model labels."),
            bullet("Contiguous geographic train, validation, and test blocks."),
            bullet("Training-only normalization statistics."),
            Spacer(1, 3 * mm),
            data_table(
                [
                    ["Dataset split", "Tiles", "Role"],
                    ["Training", "48", "Fit model weights"],
                    ["Validation", "8", "Select checkpoint"],
                    ["Test", "8", "Final held-out evaluation"],
                ],
                [45 * mm, 25 * mm, 90 * mm],
            ),
            Spacer(1, 7 * mm),
            section("2. Model and change workflow"),
            body(
                "The baseline uses an NDVI threshold of 0.6. The learned model is a six-channel U-Net with a ResNet-34 encoder and ImageNet initialization, trained for 30 CPU epochs with Dice plus binary cross-entropy loss. Epoch 30 was selected by validation IoU."
            ),
            body(
                "Clearing candidates are pixels classified as forest in 2019 and non-forest in 2024. Eight-connected components smaller than 50 pixels are removed, giving a minimum retained area of 0.5 hectares at 10-metre resolution."
            ),
            PageBreak(),
            section("3. Held-out segmentation results"),
            body(
                "Both methods were evaluated on the same valid pixels in the geographically held-out eight-tile test strip."
            ),
            Image(str(charts["model"]), width=174 * mm, height=81 * mm),
            Spacer(1, 4 * mm),
            data_table(
                [
                    ["Model", "Precision", "Recall", "F1", "IoU"],
                    ["NDVI baseline", "0.9460", "0.9608", "0.9534", "0.9109"],
                    ["U-Net", "0.9670", "0.9850", "0.9759", "0.9529"],
                ],
                [52 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm],
            ),
            Spacer(1, 6 * mm),
            body(
                "The U-Net improves IoU by 0.0420 and F1 by 0.0225. The comparison remains tied to model-generated Dynamic World labels, but geographic separation reduces leakage from adjacent tiles."
            ),
            section("4. Change-detection result"),
            body(
                "The final change mask contains 232,606 pixels across 344 retained patches, equal to 2,326.06 hectares. The corresponding Dynamic World label change contains 2,319.68 hectares in 338 patches."
            ),
            body(
                "Predicted change versus label-derived change: precision 0.8523, recall 0.8546, F1 0.8534, and IoU 0.7444."
            ),
            PageBreak(),
            section("5. Validation of all 344 patches"),
            body(
                "Each patch was compared with Dynamic World tree-label change and the independent Landsat-derived Hansen loss-year layer for 2020-2024."
            ),
            Image(str(charts["validation"]), width=174 * mm, height=76 * mm),
            Spacer(1, 4 * mm),
            data_table(
                [
                    ["Screening result", "Patches", "Area (ha)", "Area share"],
                    ["Both references", "151", "1,736.63", "74.7%"],
                    ["Dynamic World only", "95", "470.25", "20.2%"],
                    ["Hansen only", "31", "34.63", "1.5%"],
                    ["Limited or mixed", "67", "84.55", "3.6%"],
                ],
                [62 * mm, 28 * mm, 35 * mm, 35 * mm],
            ),
            Spacer(1, 6 * mm),
            body(
                "The 20 largest patches cover 1,193.00 hectares (51.3% of detected area); 19 are supported by both references. The area-weighted result is stronger than the patch count alone because most limited or mixed cases are small."
            ),
            PageBreak(),
            section("6. Largest-patch timing investigation"),
            body(
                "Patch 1 covers 173.94 hectares. Dynamic World overlaps 90.8% of the predicted patch, but Hansen support is only 18.7% when restricted to 2020-2024."
            ),
            Image(str(charts["timing"]), width=174 * mm, height=76 * mm),
            Spacer(1, 4 * mm),
            Image(str(largest_patch_image), width=174 * mm, height=50.7 * mm),
            Paragraph(
                "Figure: 2019 composite, 2024 composite, predicted clearing, and Dynamic World label change.",
                styles["SmallReport"],
            ),
            body(
                "Hansen assigns 51.8% of the patch to calendar year 2019. Because the starting composite is from July-August 2019, that annual value may include loss after the exact starting image. Combined 2019-2024 Hansen support is 70.5%. This is best interpreted as a calendar-boundary warning, not a clear false positive."
            ),
            PageBreak(),
            section("7. Limitations and responsible use"),
            bullet("Dynamic World is a model-generated consistency reference, not ground truth."),
            bullet("Hansen uses 30.92-metre pixels and calendar-year timing."),
            bullet("Hansen stand-replacement disturbance can include harvest or fire."),
            bullet("The 0.5-hectare patch filter omits smaller changes."),
            bullet("One pilot and seasonal pair do not establish wider geographic generalization."),
            bullet("Screening thresholds are deterministic review rules, not probabilities."),
            bullet("Field evidence or independent high-resolution imagery is needed for definitive attribution."),
            Spacer(1, 6 * mm),
            section("8. Conclusion"),
            body(
                "The project meets its technical objective: a reproducible geospatial ML workflow outperforms a transparent baseline, maps multi-year forest change, quantifies area, and validates every retained patch. Both references support 74.7% of detected hectares, while limited or mixed support represents 3.6%. The system is ready as a screening and prioritization tool."
            ),
            section("References"),
            Paragraph(
                "Sentinel-2 SR Harmonized: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
                styles["SmallReport"],
            ),
            Paragraph(
                "Dynamic World V1: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1",
                styles["SmallReport"],
            ),
            Paragraph(
                "Hansen Global Forest Change v1.13: https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2025_v1_13",
                styles["SmallReport"],
            ),
            Paragraph(
                "Source code: https://github.com/veerendrakosuri/deforestation-detection",
                styles["SmallReport"],
            ),
        ]
    )

    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    args = _parse_args()
    workflow = json.loads(Path(args.workflow_results).read_text(encoding="utf-8"))
    validation = json.loads(Path(args.validation_summary).read_text(encoding="utf-8"))
    temporary = Path(args.temporary_directory)
    charts = _make_charts(workflow, validation, temporary)
    _build_pdf(
        workflow,
        validation,
        Path(args.largest_patch_image),
        charts,
        Path(args.output),
    )
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
