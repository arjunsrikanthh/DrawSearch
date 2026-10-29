"""Build the DrawSearch research paper and course-length PDF write-up.

Run from the repository root with:
    python reports/build_reports.py

The PDFs use only results and source notes already checked into this project.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
SAMPLE_FIGURE = ROOT / "outputs/figures/quickdraw_samples.png"

NAVY = colors.HexColor("#10283F")
TEAL = colors.HexColor("#078985")
PALE = colors.HexColor("#E8F2F1")
INK = colors.HexColor("#233746")
MUTED = colors.HexColor("#5D7180")
GRID = colors.HexColor("#D7E1E5")

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=27, textColor=NAVY, alignment=TA_CENTER,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="PaperSubtitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10, leading=14, textColor=MUTED, alignment=TA_CENTER,
        spaceAfter=12,
    )
)
styles.add(
    ParagraphStyle(
        name="Abstract", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.1, leading=13, textColor=INK, leftIndent=18,
        rightIndent=18, spaceAfter=9,
    )
)
styles.add(
    ParagraphStyle(
        name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=14, leading=17, textColor=NAVY, spaceBefore=9, spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=13, textColor=TEAL, spaceBefore=6, spaceAfter=3,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="PaperBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.9, leading=12.4, textColor=INK, spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallNote", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.4, leading=9.6, textColor=MUTED, spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="TableHead2", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.5, leading=9, textColor=colors.white,
    )
)
styles.add(
    ParagraphStyle(
        name="TableCell2", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.4, leading=9.2, textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="WriteupTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="WriteupSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=15, textColor=NAVY, spaceBefore=6, spaceAfter=4,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="WriteupBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.5, leading=11.5, textColor=INK, spaceAfter=4,
    )
)


def P(text, style="PaperBody"):
    return Paragraph(text, styles[style])


def footer(label):
    def draw(canvas, doc):
        canvas.saveState()
        page_w, _ = letter
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.55)
        canvas.line(0.68 * inch, 0.50 * inch, page_w - 0.68 * inch, 0.50 * inch)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(0.68 * inch, 0.32 * inch, label)
        canvas.drawRightString(page_w - 0.68 * inch, 0.32 * inch, str(doc.page))
        canvas.restoreState()

    return draw


def make_results_table(compact=False):
    headers = ["Model", "Validation accuracy", "Test accuracy", "Test macro-F1", "Training time"]
    rows = [
        ["Logistic Regression", "79.13%", "78.58%", "0.7843", "1.0 s"],
        ["Linear SVM", "77.49%", "77.22%", "0.7676", "~350 s total"],
        ["Compact CNN V1", "91.34%", "91.05%", "0.9102", "17.9 s"],
        ["Compact CNN V2", "91.23%", "90.81%", "0.9076", "19.9 s"],
        ["Frozen MobileNetV3-Small", "92.16%", "91.66%", "0.9165", "642.9 s"],
    ]
    if compact:
        widths = [1.75 * inch, 1.05 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch]
    else:
        widths = [1.95 * inch, 1.1 * inch, 1.05 * inch, 0.95 * inch, 1.15 * inch]
    data = [[P(x, "TableHead2") for x in headers]]
    data += [[P(x, "TableCell2") for x in row] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_research_paper():
    path = OUT / "DrawSearch_Research_Paper.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter, rightMargin=0.68 * inch, leftMargin=0.68 * inch,
        topMargin=0.62 * inch, bottomMargin=0.68 * inch,
        title="DrawSearch: Ten-Class Doodle Classification with Classical Models and CNNs",
        author="UE24CS352A DrawSearch Project Team",
    )
    story = [
        P("DrawSearch: Ten-Class Doodle Classification<br/>with Classical Models and Compact CNNs", "PaperTitle"),
        P("UE24CS352A Machine Learning Mini-Project | PES University | October 2026", "PaperSubtitle"),
        P(
            "<b>Abstract.</b> This project studies closed-set classification of small, hand-drawn images. "
            "Using ten categories from Google Quick, Draw!, we prepared a fixed, balanced dataset of "
            "100,000 28 x 28 grayscale drawings and compared Logistic Regression, a linear support "
            "vector machine, two compact convolutional neural networks (CNNs), and a frozen "
            "MobileNetV3-Small feature extractor. All models use the same train/validation/test split; "
            "validation data selects model settings and checkpoints, and test data is reserved for final "
            "scoring. The compact CNN V1 reached 91.05% test accuracy and 0.9102 macro-F1, compared "
            "with 78.58% and 0.7843 for Logistic Regression. Frozen MobileNetV3-Small achieved the "
            "highest measured test accuracy (91.66%) and macro-F1 (0.9165), at higher inference cost "
            "and with a larger checkpoint. The user-facing Gradio demo uses CNN V1 to classify a drawing "
            "into one of the ten labels. Results are limited to one balanced split and do not establish "
            "performance on people-disjoint data or arbitrary real-world sketches.",
            "Abstract",
        ),
        P("1. Introduction", "SectionTitle"),
        P(
            "Hand-drawn marks are compact but highly variable: people simplify shape, choose different "
            "viewpoints, and omit details. This project asks a bounded question: can a lightweight local "
            "classifier map one 28 x 28 doodle to one of ten chosen object categories? The labels are "
            "cat, dog, bicycle, car, airplane, banana, apple, clock, house, and tree. The system reports "
            "ranked class scores through a drawing canvas. It is a closed-set classification demo; it does "
            "not retrieve photographs or perform sketch-to-image search.",
        ),
        P(
            "The work is inspired by the Stanford CS229 report <i>Drawing: A New Way To Search</i> [1], "
            "which compares classical and neural approaches to doodle recognition. We reimplement a "
            "small subset of model ideas in a current PyTorch pipeline. The historical project uses a "
            "different set of categories, data protocol, and implementation; its scores are context only "
            "and are not a directly comparable benchmark.",
        ),
        P("2. Dataset and Experimental Protocol", "SectionTitle"),
        P("2.1 Data source and sampling", "Subsection"),
        P(
            "The data comes from Google's Quick, Draw! dataset. The supplied simplified bitmap files "
            "store drawings as 28 x 28 grayscale images [2]. For each of the ten selected classes, we "
            "sampled 10,000 distinct source rows using seed 42, giving 100,000 examples. Each class is "
            "split into 8,000 training, 1,000 validation, and 1,000 test images. The resulting partitions "
            "contain 80,000, 10,000, and 10,000 images respectively and are balanced by class. The "
            "tracked manifest records class order, source checksums, source row IDs, and split hashes. "
            "The preparation audit reports no exact duplicate bitmaps shared across partitions.",
        ),
        P("2.2 Data use and split discipline", "Subsection"),
        P(
            "Pixel values are scaled to [0, 1] for the pixel-based models. Every experiment refers to "
            "the same split manifest. Hyperparameter selection and checkpoint selection use training "
            "and validation data; the held-out test partition is used only for final reporting. The "
            "prepared split and raw downloads are excluded from Git because of their size. The compact "
            "manifest and evaluation records are tracked to preserve provenance without committing "
            "the full data.",
        ),
        P("3. Models and Implementation", "SectionTitle"),
        P("3.1 Classical baselines", "Subsection"),
        P(
            "Logistic Regression and LinearSVC receive flattened 784-element vectors. Logistic "
            "Regression uses the lbfgs solver with a maximum of 200 iterations. For LinearSVC, the "
            "candidate regularization values C = 0.01 and C = 0.1 are compared on validation accuracy; "
            "macro-F1 and then smaller C break ties. The selected C = 0.1 model is refit on training plus "
            "validation data before its single final test evaluation. No RBF kernel experiment was run.",
        ),
        P("3.2 Compact convolutional networks", "Subsection"),
        P(
            "CNN V1 uses two convolution-and-pooling stages followed by a dense classifier. CNN V2 "
            "uses a one-convolution alternative. Both output ten class scores and were trained with the "
            "same training partition. Checkpoints were chosen using validation data. CNN V1 is the demo "
            "model because it combines strong measured accuracy with a compact checkpoint and fast "
            "single-drawing inference on the recorded Apple MPS setup.",
        ),
        P("3.3 Transfer-learning comparison", "Subsection"),
        P(
            "The transfer experiment repeats each grayscale bitmap across three channels, resizes it to "
            "224 x 224, and applies ImageNet normalization. The MobileNetV3-Small feature extractor is "
            "frozen and only a newly initialized classification head is trained for five epochs. The "
            "validation-accuracy checkpoint is used for test scoring. This model is included as an "
            "experiment, not as the live demo backend.",
        ),
        P("3.4 Canvas inference path", "Subsection"),
        P(
            "The local Gradio app reads the trained CNN V1 checkpoint and the saved label order. "
            "Incoming strokes are cropped, resized, and centered to form a 28 x 28 input, then "
            "normalized to the CNN's training range. The interface displays the leading class scores. "
            "A single hand-drawn house passed a recorded end-to-end smoke check. That check confirms "
            "the path runs; it is not a user study or a general estimate of live-drawing accuracy.",
        ),
        P("4. Evaluation", "SectionTitle"),
        P(
            "Accuracy is the fraction of correct predictions. Macro-F1 is the unweighted mean of the "
            "ten per-class F1 scores, so each class contributes equally. Table 1 reports the saved "
            "validation accuracy, held-out test accuracy, macro-F1, and recorded training time. Times "
            "were measured on an Apple silicon machine using MPS where applicable; they are hardware "
            "specific and should not be interpreted as cross-device performance guarantees.",
        ),
        make_results_table(),
        Spacer(1, 3),
        P("Table 1. Results on the same fixed, balanced split. Linear SVM time includes candidate fits and final refit.", "SmallNote"),
        P("5. Results and Discussion", "SectionTitle"),
        P(
            "CNN V1 improved test accuracy by 12.47 percentage points over Logistic Regression on this "
            "split, while CNN V2 was 0.24 points below V1. Frozen MobileNetV3-Small scored 0.61 points "
            "above CNN V1 in test accuracy. Its recorded single-drawing latency was 4.11 ms versus "
            "0.158 ms for CNN V1, and its checkpoint was about 3.87 MB versus 0.19 MB. These are "
            "measurements from the saved experiment records, not controlled latency benchmarks across "
            "devices. The small accuracy difference alone does not establish that MobileNet is "
            "statistically better; no repeated-seed or significance analysis was performed.",
        ),
        P(
            "The two linear baselines are substantially below the convolutional models. This is "
            "consistent with local filters being useful for spatial stroke patterns, but the comparison "
            "does not isolate architecture as the sole cause: the model families have different "
            "optimization procedures and training budgets. Per-class records show that some categories "
            "are easier than others; the dog class is among the weaker classes for the pixel-based "
            "baselines, while small CNNs raise performance across the class-balanced evaluation. The "
            "full per-class precision, recall, and F1 values remain in the tracked JSON files.",
        ),
        P("6. Limitations and Threats to Validity", "SectionTitle"),
        P(
            "<b>Scope:</b> the classifier always chooses among ten labels. It has no reject option for "
            "blank, ambiguous, or out-of-scope drawings. The live interface is a local demonstration, "
            "not a general-purpose search system. <b>Sampling:</b> evaluation uses one fixed random "
            "split and one reported training run per model. Quick, Draw! records in these bitmap files "
            "do not provide a person-disjoint split for this experiment, so test examples may share "
            "population-level drawing conventions with training data. <b>External validity:</b> results "
            "on Google game drawings may not transfer to sketches made with other tools or by different "
            "populations. <b>Comparison:</b> the Stanford paper's category set, data size, and validation "
            "protocol differ. Its reported scores are not a head-to-head baseline. <b>Unmeasured work:</b> "
            "no RBF SVM, CNN V3/V4, repeated random seeds, calibration, or human-subject evaluation was "
            "completed.",
        ),
        P("7. Conclusion", "SectionTitle"),
        P(
            "This project built and evaluated a reproducible ten-class doodle classifier with a local "
            "canvas demo. The fixed-split comparison found that compact CNN V1 substantially outperformed "
            "the two flattened-pixel baselines and remained close to the best measured test result while "
            "being smaller and faster in the recorded local inference measurements. The outcome is a "
            "working classifier and a documented comparison, not evidence of broad sketch retrieval or "
            "performance outside the selected classes. Further work should measure repeated seeds, "
            "person-disjoint generalization, uncertainty and out-of-class rejection, and user-drawn "
            "examples collected under a defined evaluation protocol.",
        ),
        P("References", "SectionTitle"),
        P(
            "[1] N. M. Phu, C. Xiao, and J. Muindi, “Drawing: A New Way To Search,” Stanford CS229 "
            "project report, 2018. https://cs229.stanford.edu/proj2018/report/33.pdf<br/>"
            "[2] Google Creative Lab, “Quick, Draw! Dataset,” source code and dataset documentation. "
            "https://github.com/googlecreativelab/quickdraw-dataset. Dataset license: CC BY 4.0, "
            "https://creativecommons.org/licenses/by/4.0/.<br/>"
            "[3] DrawSearch project implementation, experiment records, and data manifest. "
            "https://github.com/arjunsrikanthh/DrawSearch (private repository).",
            "SmallNote",
        ),
    ]
    doc.build(story, onFirstPage=footer("DrawSearch | Research-style project report"), onLaterPages=footer("DrawSearch | Research-style project report"))
    return path


def build_course_writeup():
    path = OUT / "DrawSearch_Course_Writeup.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=letter, rightMargin=0.68 * inch, leftMargin=0.68 * inch,
        topMargin=0.52 * inch, bottomMargin=0.66 * inch,
        title="DrawSearch: UE24CS352A Mini-Project Write-up",
        author="UE24CS352A DrawSearch Project Team",
    )
    story = [
        P("DrawSearch", "WriteupTitle"),
        P("Ten-class hand-drawn object classification | UE24CS352A Machine Learning mini-project", "PaperSubtitle"),
        P("Problem statement", "WriteupSection"),
        P(
            "DrawSearch classifies a 28 x 28 drawing into one of ten fixed Quick, Draw! categories: "
            "cat, dog, bicycle, car, airplane, banana, apple, clock, house, or tree. A local drawing "
            "canvas lets a user draw an object and view the model's leading guesses. This is a closed-set "
            "image-classification project; it does not search for photographs or perform sketch-to-image retrieval.",
            "WriteupBody",
        ),
        P("Dataset details", "WriteupSection"),
        P(
            "The source is Google's Quick, Draw! dataset of simplified 28 x 28 grayscale bitmap images. "
            "We selected 10,000 distinct rows from each of ten classes (100,000 examples total) with "
            "seed 42. Each class has 8,000 training, 1,000 validation, and 1,000 test examples. The "
            "saved manifest records the class order, source checksums, selected source row IDs, and split "
            "hashes. The split audit reports no exact duplicate bitmap across the three partitions. "
            "The dataset is credited to Google and licensed CC BY 4.0.",
            "WriteupBody",
        ),
        P("Approach and implementation", "WriteupSection"),
        P(
            "We compared Logistic Regression and LinearSVC on flattened pixel vectors with two compact "
            "CNNs that learn local stroke patterns. A frozen MobileNetV3-Small model was also evaluated "
            "as a transfer-learning comparison. All experiments use the same fixed split. Settings and "
            "checkpoints were selected using training and validation data; the held-out test split was "
            "used only for final scores. The local Gradio app loads compact CNN V1, preprocesses canvas "
            "strokes to 28 x 28, and displays the highest scoring labels. README instructions explain "
            "environment setup, data preparation, training, and app launch.",
            "WriteupBody",
        ),
        make_results_table(compact=True),
        Spacer(1, 3),
        P("Table 1. Held-out results on the same 10,000-image test split.", "SmallNote"),
        P("Conclusion", "WriteupSection"),
        P(
            "Compact CNN V1 achieved 91.05% test accuracy and 0.9102 macro-F1, higher than the "
            "Logistic Regression and linear SVM baselines. Frozen MobileNetV3-Small had the highest "
            "measured test accuracy (91.66%), but a larger checkpoint and higher recorded inference "
            "latency; CNN V1 therefore remains the demo model. These results apply to one balanced split "
            "and ten labels. The classifier always predicts one of those labels, even for an unrelated "
            "or blank drawing. Broader user-drawing performance and repeated-seed stability have not been "
            "measured.",
            "WriteupBody",
        ),
        P(
            "References: Phu, Xiao, and Muindi, “Drawing: A New Way To Search,” Stanford CS229, 2018, "
            "https://cs229.stanford.edu/proj2018/report/33.pdf. Google Creative Lab, Quick, Draw! "
            "Dataset, https://github.com/googlecreativelab/quickdraw-dataset (CC BY 4.0).",
            "SmallNote",
        ),
    ]
    doc.build(story, onFirstPage=footer("DrawSearch | Course write-up"), onLaterPages=footer("DrawSearch | Course write-up"))
    return path


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print(build_research_paper())
    print(build_course_writeup())
