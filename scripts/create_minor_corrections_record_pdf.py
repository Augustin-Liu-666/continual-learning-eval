from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


OUT_PATH = Path(__file__).resolve().parents[1] / "reports" / "Record_of_Minor_Corrections_Changlin_Liu.pdf"


rows = [
    [
        "Examiner comment / required correction",
        "Correction made",
        "Location in revised thesis",
    ],
    [
        'Use "continual learning" consistently instead of "continuous learning".',
        'The terminology was standardised throughout the thesis. A final search found no remaining uses of "continuous learning".',
        "Throughout the thesis, including the title, Abstract, Chapter 1, Chapter 2, and Conclusion.",
    ],
    [
        "Standardise technical terms such as backward transfer, average forgetting, episodic memory, replay buffer, and Averaged Gradient Episodic Memory.",
        "Key continual-learning terms were standardised. Incorrect or inconsistent expressions were replaced with the accepted technical terms.",
        "Sections 4.3, 4.5, 5.5, 6.2, 6.4, Chapter 7, and Chapter 8.",
    ],
    [
        'Correct incorrect expressions such as "backward transmission", "positive transmission", "plot memory", "occasional memory", and "A-GERM".',
        'These expressions were corrected or removed. The thesis now uses "backward transfer", "positive backward transfer", "episodic memory", "replay buffer", and "A-GEM" as appropriate.',
        "Throughout the revised thesis, especially Sections 4.3, 4.5.6, 6.2.2, 6.4, and 7.2.",
    ],
    [
        "Remove repeated discussion of the EMNIST results in Section 6.2.2.",
        "The EMNIST discussion was revised to avoid repetition and to focus on the interpretation of positive backward transfer.",
        "Section 6.2.2 and related discussion in Chapter 7.",
    ],
    [
        "Correct duplicated wording on the title page.",
        'The title page was corrected so that the degree wording reads "Master of Science in Computer Science" without repeating "Computer Science".',
        "Title page.",
    ],
    [
        "Clarify the memory-budget difference between ER and A-GEM.",
        "The thesis now explicitly states that ER uses a fixed replay buffer, while A-GEM accumulates episodic memory per task in this implementation. The comparison is described as not having a strictly matched total memory budget.",
        "Sections 3.4, 4.5.6, 5.5, 6.5, 7.2, 7.4, 8.2, and 8.4.",
    ],
    [
        "Acknowledge limitations relating to three random seeds and absence of formal significance testing.",
        "The revised thesis states that experiments used three random seeds and that further runs and formal significance testing would strengthen the reliability of the conclusions.",
        "Sections 4.6, 5.5, 7.4, and 8.4.",
    ],
    [
        "Avoid over-generalising replay as always superior.",
        "The discussion and conclusion were revised to state that ER is strongest in forgetting-dominated settings, while EMNIST shows positive transfer and replay is not always necessary.",
        "Sections 6.2.2, 7.1, 7.3, 8.1, 8.2, and 8.5.",
    ],
    [
        "Improve referencing, typographical errors, grammar, capitalisation, and presentation.",
        "A proofreading pass was completed. Bibliography spacing and formatting issues were corrected, including missing spaces around conference and journal names.",
        "Bibliography and throughout the revised thesis.",
    ],
    [
        "Remove examination/confidentiality wording for the final clean version.",
        "The final title page no longer contains the statement that the thesis is for examination purposes only or confidential to the examination process.",
        "Title page.",
    ],
]


def para(text, style):
    return Paragraph(text.replace("&", "&amp;"), style)


styles = getSampleStyleSheet()
normal = ParagraphStyle(
    "NormalSmall",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.6,
    leading=10.5,
    spaceAfter=0,
)
header = ParagraphStyle(
    "HeaderSmall",
    parent=normal,
    fontName="Helvetica-Bold",
)
title = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    alignment=1,
    spaceAfter=12,
)
meta = ParagraphStyle(
    "Meta",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=13,
    spaceAfter=3,
)

doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=A4,
    rightMargin=15 * mm,
    leftMargin=15 * mm,
    topMargin=15 * mm,
    bottomMargin=15 * mm,
)

story = [
    Paragraph("Record of Minor Corrections", title),
    Paragraph("Candidate: Changlin Liu", meta),
    Paragraph("Thesis title: An Empirical Evaluation of Continual Learning under Visual Drift", meta),
    Paragraph("Degree: Master of Science in Computer Science", meta),
    Paragraph("Date: 6 August 2026", meta),
    Spacer(1, 6),
    Paragraph(
        "This document records the minor corrections made in response to the examiners' comments before submission of the revised clean thesis.",
        meta,
    ),
    Spacer(1, 8),
]

table_data = []
for i, row in enumerate(rows):
    style = header if i == 0 else normal
    table_data.append([para(cell, style) for cell in row])

table = Table(table_data, colWidths=[58 * mm, 68 * mm, 39 * mm], repeatRows=1)
table.setStyle(
    TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story.append(table)

doc.build(story)
print(OUT_PATH)
