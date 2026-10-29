# DrawSearch

DrawSearch classifies a drawing as one of ten fixed Quick, Draw! categories. It
includes classical machine-learning baselines, two compact CNNs, a MobileNet
transfer-learning experiment, and a local drawing-canvas demo. It does not
search for photographs or perform sketch-to-image retrieval.

## Start here

1. Read [PROJECT_DETAILS.md](PROJECT_DETAILS.md) for the project story, design,
   data, experiments, limitations, and file guide.
2. Install Python 3.12 and create an environment:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Run the checks, prepare data, train the baseline models, and launch the app:

   ```bash
   python -m unittest discover -s tests -v
   python scripts/prepare_data.py
   python scripts/train_models.py --epochs 10
   python scripts/run_app.py
   ```

The first data preparation downloads around 1.2 GB of source drawings. It
creates ignored data archives locally; model checkpoints and generated plots
are also excluded from Git. The app prints its local address, usually
`http://127.0.0.1:7860`.

The checked-in `outputs/metrics/` JSON files record the completed experiments.
They describe measured results; trained model files are intentionally not
included. To run the optional linear SVM and MobileNet experiments, see the
commands and weight-download note in [PROJECT_DETAILS.md](PROJECT_DETAILS.md).

## Scope at a glance

- Ten classes: cat, dog, bicycle, car, airplane, banana, apple, clock, house,
  and tree.
- The user-facing demo loads the small CNN V1 checkpoint after it is trained.
- RBF SVM and CNN versions 3 and 4 have not been implemented or measured.

## Course deliverables

- [Research-style project report](reports/DrawSearch_Research_Paper.pdf) covers
  the dataset, methods, results, limitations, and references.
- [Course write-up](reports/DrawSearch_Course_Writeup.pdf) is a concise, one-page
  PDF covering the problem, dataset, approach, implementation, and conclusion.
- [Presentation slides](presentation/DrawSearch_Slides_v3.pptx) are the current
  editable deck for the project review.
- [Report builder](reports/build_reports.py) regenerates both PDFs from the
  project results. Run `python reports/build_reports.py` from the repository
  root after installing the project requirements.

The prepared image split and trained checkpoints remain outside Git because of
their size. A new user must download data and train a checkpoint to reproduce
the full demo. The report and slides summarize saved experiment results; they
do not replace the live code demonstration.

Quick, Draw! source drawings are from Google's dataset under CC BY 4.0. See
the project details for dataset provenance, paper context, and references.
