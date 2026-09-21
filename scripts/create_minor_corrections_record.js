const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  AlignmentType,
  HeadingLevel,
} = require("docx");

const outPath = path.join(
  __dirname,
  "..",
  "reports",
  "Record_of_Minor_Corrections_Changlin_Liu.docx"
);

const border = { style: BorderStyle.SINGLE, size: 1, color: "B7B7B7" };
const borders = { top: border, bottom: border, left: border, right: border };

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0 },
    alignment: opts.alignment,
    children: [
      new TextRun({
        text,
        bold: opts.bold,
        italics: opts.italics,
        size: opts.size,
      }),
    ],
  });
}

function cell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    margins: { top: 120, bottom: 120, left: 120, right: 120 },
    children: [
      new Paragraph({
        spacing: { after: 0 },
        children: [new TextRun({ text, bold: opts.header, size: 21 })],
      }),
    ],
  });
}

const rows = [
  [
    "Examiner comment / required correction",
    "Correction made",
    "Location in revised thesis",
  ],
  [
    "Use \"continual learning\" consistently instead of \"continuous learning\".",
    "The terminology was standardised throughout the thesis. A final search found no remaining uses of \"continuous learning\".",
    "Throughout the thesis, including the title, Abstract, Chapter 1, Chapter 2, and Conclusion.",
  ],
  [
    "Standardise technical terms such as backward transfer, average forgetting, episodic memory, replay buffer, and Averaged Gradient Episodic Memory.",
    "Key continual-learning terms were standardised. Incorrect or inconsistent expressions were replaced with the accepted technical terms.",
    "Sections 4.3, 4.5, 5.5, 6.2, 6.4, Chapter 7, and Chapter 8.",
  ],
  [
    "Correct incorrect expressions such as \"backward transmission\", \"positive transmission\", \"plot memory\", \"occasional memory\", and \"A-GERM\".",
    "These expressions were corrected or removed. The thesis now uses \"backward transfer\", \"positive backward transfer\", \"episodic memory\", \"replay buffer\", and \"A-GEM\" as appropriate.",
    "Throughout the revised thesis, especially Sections 4.3, 4.5.6, 6.2.2, 6.4, and 7.2.",
  ],
  [
    "Remove repeated discussion of the EMNIST results in Section 6.2.2.",
    "The EMNIST discussion was revised to avoid repetition and to focus on the interpretation of positive backward transfer.",
    "Section 6.2.2 and related discussion in Chapter 7.",
  ],
  [
    "Correct duplicated wording on the title page.",
    "The title page was corrected so that the degree wording reads \"Master of Science in Computer Science\" without repeating \"Computer Science\".",
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
];

const tableRows = rows.map((row, index) =>
  new TableRow({
    tableHeader: index === 0,
    children: [
      cell(row[0], 3300, { header: index === 0 }),
      cell(row[1], 3600, { header: index === 0 }),
      cell(row[2], 2460, { header: index === 0 }),
    ],
  })
);

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", bold: true, size: 32 },
        paragraph: { spacing: { before: 200, after: 160 }, outlineLevel: 0 },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          alignment: AlignmentType.CENTER,
          children: [new TextRun("Record of Minor Corrections")],
        }),
        p("Candidate: Changlin Liu", { after: 80 }),
        p("Thesis title: An Empirical Evaluation of Continual Learning under Visual Drift", { after: 80 }),
        p("Degree: Master of Science in Computer Science", { after: 80 }),
        p("Date: 6 August 2026", { after: 240 }),
        p(
          "This document records the minor corrections made in response to the examiners' comments before submission of the revised clean thesis.",
          { after: 240 }
        ),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [3300, 3600, 2460],
          rows: tableRows,
        }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, buffer);
  console.log(outPath);
});
