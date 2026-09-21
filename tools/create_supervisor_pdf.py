from pathlib import Path
import argparse
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "reports" / "supervisor_progress_update_zh.md"
DEFAULT_OUTPUT = ROOT / "reports" / "supervisor_progress_update_zh.pdf"

FIGURES = [
    ("Overall cross-method thesis summary", ROOT / "reports" / "thesis_summary.png"),
    ("All-dataset accuracy trends", ROOT / "reports" / "linechart_all_datasets.png"),
    ("Buffer/memory sensitivity by dataset", ROOT / "reports" / "buffer_sensitivity_by_dataset.png"),
    ("Accuracy-cost trade-off", ROOT / "reports" / "cost_vs_acc_scatter.png"),
    ("Rotated MNIST multi-seed comparison", ROOT / "reports" / "comparison_multiseed_rotated_mnist.png"),
    ("CIFAR-100 multi-seed comparison", ROOT / "reports" / "comparison_multiseed_cifar-100.png"),
]

ACTIVE_FONT = "STSong-Light"


def register_fonts(font_choice="cjk"):
    global ACTIVE_FONT
    fonts = {
        "times": {
            "normal": "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "bold": "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "name": "TimesNewRoman",
            "bold_name": "TimesNewRoman-Bold",
        },
        "arial": {
            "normal": "/System/Library/Fonts/Supplemental/Arial.ttf",
            "bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "name": "Arial",
            "bold_name": "Arial-Bold",
        },
        "verdana": {
            "normal": "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "bold": "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
            "name": "Verdana",
            "bold_name": "Verdana-Bold",
        },
    }

    if font_choice in fonts and Path(fonts[font_choice]["normal"]).exists():
        spec = fonts[font_choice]
        pdfmetrics.registerFont(TTFont(spec["name"], spec["normal"]))
        pdfmetrics.registerFont(TTFont(spec["bold_name"], spec["bold"]))
        pdfmetrics.registerFontFamily(
            spec["name"],
            normal=spec["name"],
            bold=spec["bold_name"],
            italic=spec["name"],
            boldItalic=spec["bold_name"],
        )
        ACTIVE_FONT = spec["name"]
        return spec["name"]

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    ACTIVE_FONT = "STSong-Light"
    return "STSong-Light"


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def build_table(lines, styles):
    rows = []
    for line in lines:
        if is_table_separator(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append([Paragraph(clean_inline(c), styles["TableCell"]) for c in cells])

    if not rows:
        return []

    col_count = max(len(r) for r in rows)
    for row in rows:
        while len(row) < col_count:
            row.append(Paragraph("", styles["TableCell"]))

    page_width = A4[0] - 3.2 * cm
    col_widths = [page_width / col_count] * col_count
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), ACTIVE_FONT),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [Spacer(1, 5), table, Spacer(1, 9)]


def parse_markdown(md_text: str, styles):
    story = []
    lines = md_text.splitlines()
    i = 0
    para = []

    def flush_para():
        if para:
            story.append(Paragraph(clean_inline(" ".join(para)), styles["Body"]))
            story.append(Spacer(1, 6))
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            flush_para()
            i += 1
            continue

        if line.startswith("|"):
            flush_para()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.extend(build_table(table_lines, styles))
            continue

        if line.startswith("#"):
            flush_para()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            style_name = "Title" if level == 1 else "Heading2" if level == 2 else "Heading3"
            story.append(Paragraph(clean_inline(text), styles[style_name]))
            story.append(Spacer(1, 8 if level <= 2 else 5))
            i += 1
            continue

        if line.strip() == "---":
            flush_para()
            story.append(Spacer(1, 8))
            i += 1
            continue

        bullet_match = re.match(r"^(\s*)-\s+(.+)$", line)
        numbered_match = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if bullet_match:
            flush_para()
            story.append(
                Paragraph(clean_inline(bullet_match.group(2)), styles["Bullet"], bulletText="•")
            )
            i += 1
            continue
        if numbered_match:
            flush_para()
            story.append(Paragraph(clean_inline(numbered_match.group(1)), styles["Bullet"]))
            i += 1
            continue

        para.append(line.strip())
        i += 1

    flush_para()
    return story


def add_figures(story, styles):
    story.append(PageBreak())
    story.append(Paragraph("Key Experimental Figures", styles["Title"]))
    story.append(Spacer(1, 10))

    max_width = A4[0] - 3.2 * cm
    max_height = A4[1] - 5.2 * cm

    for idx, (caption, path) in enumerate(FIGURES, 1):
        if not path.exists():
            continue
        img = Image(str(path))
        scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale

        block = [
            Paragraph(f"Figure {idx}. {clean_inline(caption)}", styles["FigureCaption"]),
            Spacer(1, 8),
            img,
        ]
        story.append(KeepTogether(block))
        if idx != len(FIGURES):
            story.append(PageBreak())


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(ACTIVE_FONT, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font", choices=["cjk", "times", "arial", "verdana"], default="cjk")
    args = parser.parse_args()

    font = register_fonts(args.font)
    sample = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName=font,
            fontSize=18,
            leading=24,
            textColor=colors.HexColor("#111827"),
            spaceAfter=8,
            alignment=TA_CENTER,
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=sample["Heading2"],
            fontName=font,
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "Heading3": ParagraphStyle(
            "Heading3",
            parent=sample["Heading3"],
            fontName=font,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#334155"),
            spaceBefore=4,
            spaceAfter=3,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#111827"),
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=10,
            leading=15,
            leftIndent=14,
            firstLineIndent=-8,
            textColor=colors.HexColor("#111827"),
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=7.2,
            leading=9,
            wordWrap="CJK",
        ),
        "FigureCaption": ParagraphStyle(
            "FigureCaption",
            parent=sample["BodyText"],
            fontName=font,
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1F2937"),
        ),
    }

    story = parse_markdown(args.source.read_text(encoding="utf-8"), styles)
    add_figures(story, styles)

    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Supervisor Progress Update",
        author="Lafayette Antonio",
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(args.output)


if __name__ == "__main__":
    main()
