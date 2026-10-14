# DrawSearch: project details

## 1. The question and the scope

This project asks whether a small computer-vision model can recognize a simple
hand-drawn object. The input is one drawing; the output is a ranked guess from
ten fixed labels: **cat, dog, bicycle, car, airplane, banana, apple, clock,
house, and tree**. The local demo is a classifier. It does not search the web,
retrieve photographs, or turn sketches into photos.

The project is inspired by the Stanford CS229 student paper *Drawing: A New
Way To Search*. The paper motivates comparisons among linear models, SVMs,
CNNs, and transfer learning. Its historical dataset, class selection, and
implementation differ from this project, so its reported scores are not
directly comparable to the results below. The paper and its authors' code were
used as background; this repository is a modern PyTorch implementation.

## 2. Data, labels, and reproducibility

The source is Google's Quick, Draw! NumPy bitmap dataset. Each source image is
a 28 by 28 grayscale drawing. The preparation script downloads one source
file for each selected class, then chooses 10,000 distinct source rows per
class using seed 42. It partitions those rows into 8,000 training, 1,000
validation, and 1,000 test examples per class. The resulting dataset has
100,000 images: 80,000 for training, 10,000 for validation, and 10,000 for
testing. Each split has equal class counts.

`configs/data.yaml` defines the class order, seed, and counts. The same class
order is used for numeric labels, checkpoints, metrics, and the user
interface. `data/processed/manifest.json` records source checksums, selected
source row IDs, split checksums, and class order. `inspection.json` records
pixel and duplicate checks. The manifest is tracked; the downloaded source
files and generated `.npz` split archives are ignored by Git because they are
large. A fresh checkout therefore needs to run the data preparation command
before training. Preparation downloads about 1.2 GB of source files.

The saved experiments share a manifest hash, so their reported results can
be tied to the same fixed split. The test split is used for final scoring only
after model and checkpoint choices are made using the training and validation
data.

## 3. Models and how they work

The experiments progress from simple baselines to image models:

1. **Logistic Regression** flattens each 28 by 28 image into 784 pixel values
   and predicts one of the ten labels. It provides a quick classical baseline.
2. **Linear SVM** uses the same flattened pixels. The tested `C` values are
   compared on validation data; the selected model is refit on training plus
   validation data before its one final test evaluation.
3. **CNN V1** uses two convolution and pooling stages to learn local stroke
   patterns, followed by a dense layer and ten output scores. This is the
   model loaded by the drawing demo because it is compact and fast.
4. **CNN V2** is a smaller-depth comparison with one convolution and pooling
   stage, followed by a dense layer and ten outputs.
5. **Frozen MobileNetV3-Small** repeats each grayscale image into three
   channels, resizes it to 224 by 224, and uses ImageNet normalization. Its
   pretrained image feature extractor is frozen; only a new small classifier
   head is trained. It is an experiment and is not used by the demo.

Accuracy is the fraction of correctly classified test drawings. Macro-F1 is
the mean F1 score across the ten classes, giving each class equal weight.
Results from the saved JSON records are:

| Model | Test accuracy | Test macro-F1 | Notes |
|---|---:|---:|---|
| Logistic Regression | 78.58% | 0.7843 | Flattened pixels |
| Linear SVM | 77.22% | 0.7676 | Validation-selected linear model |
| CNN V1 | 91.05% | 0.9102 | Demo model; compact checkpoint |
| CNN V2 | 90.81% | 0.9076 | One-convolution variant |
| Frozen MobileNetV3-Small | 91.66% | 0.9165 | Highest score here; larger and slower locally |

These values describe this particular split and implementation. They do not
show how the models perform on every kind of drawing or on classes outside
the ten labels. No RBF-kernel SVM or CNN versions 3 and 4 were implemented or
measured. The linear SVM and these two CNN variants are the completed
comparisons in this repository.

## 4. From a browser drawing to a prediction

The app is a local Gradio page. Its path through the code is:

1. `scripts/run_app.py` starts the application.
2. `drawsearch/app.py` loads `outputs/models/compact_cnn.pt` and reads the
   label order from the dataset manifest. It checks that checkpoint labels
   and manifest labels agree.
3. `drawsearch/preprocessing.py` crops the strokes, rescales and centers the
   drawing, and converts it to a 28 by 28 white-stroke-on-black image.
4. The image is normalized to the same 0-to-1 pixel range used in CNN
   training. The model returns scores, which the app displays as the top
   guesses.

The app requires locally generated data metadata and a trained CNN V1
checkpoint. A previous local smoke check loaded the saved model and
classified a drawn house as `house`; the running Gradio page returned HTTP
200. This is a smoke check, not a broad user study or a guarantee of accuracy
on arbitrary user drawings.

## 5. Code and data file guide

| Path | What it is for |
|---|---|
| `README.md` | Short entry point and setup commands |
| `PROJECT_DETAILS.md` | Full project explanation and experiment record |
| `configs/data.yaml` | Dataset labels, counts, dimensions, and seed |
| `drawsearch/data.py` | Downloads source data, creates splits and audit records |
| `drawsearch/models.py` | Compact CNN V1 and V2 architectures |
| `drawsearch/transfer_model.py` | Isolated MobileNet transfer-learning model |
| `drawsearch/preprocessing.py` | Converts canvas strokes to model input |
| `drawsearch/app.py` | Loads CNN V1 and builds the drawing interface |
| `scripts/prepare_data.py` | Command-line entry point for dataset preparation |
| `scripts/train_models.py` | Trains Logistic Regression and CNN V1 |
| `scripts/train_svm.py` | Runs linear SVM selection and final scoring |
| `scripts/train_cnn_v2.py` | Trains and scores CNN V2 |
| `scripts/train_transfer_learning.py` | Trains and scores frozen MobileNet head |
| `scripts/transfer_learning_smoke_test.py` | Small transfer-model wiring check |
| `scripts/check_environment.py` | Checks required software and local setup |
| `scripts/run_app.py` | Starts the local Gradio app |
| `tests/` | Unit tests for data splitting, model shapes, and canvas preprocessing |
| `data/processed/manifest.json` | Tracked provenance and fixed-split record |
| `data/processed/inspection.json` | Tracked dataset inspection summary |
| `outputs/metrics/` | Tracked machine-readable baseline, CNN, and SVM results |
| `outputs/transfer_learning/metrics.json` | Tracked MobileNet experiment results |
| `outputs/figures/quickdraw_samples.png` | Small visual sample of the selected classes |

Large source and split data, model checkpoints, environment-specific setup
records, confusion-matrix images, presentations, and PDFs are not tracked.
The metrics JSON files retain the detailed per-class results and experiment
settings without needing to include trained weights.

## 6. Setup and running the work

The dependency versions in `requirements.txt` were used with Python 3.12 on
macOS Apple silicon. Other systems may need compatible dependency versions;
Windows commands are not verified here. From the repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python scripts/prepare_data.py
```

Training the baseline and demo model, then starting the app:

```bash
python scripts/train_models.py --epochs 10
python scripts/run_app.py
```

Optional experiment commands:

```bash
python scripts/train_svm.py --c-values 0.01,0.1
python scripts/train_cnn_v2.py --epochs 10
python scripts/transfer_learning_smoke_test.py
```

The full MobileNet run requires pretrained weights. The program will not
silently fetch them. To explicitly download the torchvision default weights
into the local PyTorch cache, run:

```bash
python - <<'PY'
from pathlib import Path
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
from torchvision.models import MobileNet_V3_Small_Weights

weights = MobileNet_V3_Small_Weights.DEFAULT
destination = Path(get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
destination.parent.mkdir(parents=True, exist_ok=True)
download_url_to_file(weights.url, destination, hash_prefix="047dcff4")
print(destination)
PY
python scripts/train_transfer_learning.py --train --pretrained --epochs 5 --batch-size 64 --device auto
```

The full training command saves checkpoints under ignored `outputs/models/`
and writes result records under `outputs/`. Running `prepare_data.py` again
when split archives already exist is refused unless `--force` is supplied.
Use `--force` only when deliberately rebuilding the local fixed split.

## 7. Verification and limits

The completed local review included eight passing unit tests, Python source
compilation, dependency consistency checks, a transfer-learning smoke test,
the saved-model drawing smoke check, and an HTTP check of the local app. The
full training metrics are preserved in the JSON files. These checks were run
on the original Mac environment; they are not claims that all operating
systems or arbitrary installations have been validated.

The project is intentionally scoped to ten classes and a single local
classifier. The prepared image split and checkpoints are not delivered in the
Git repository, so a new user must download data and train a checkpoint to
reproduce the full demo. No PDF report or slide deck is included here.

## References and data credit

- Phu, Nguyet Minh; Xiao, Connie; Muindi, Jervis. *Drawing: A New Way To
  Search*, Stanford CS229, 2018. [Paper](https://cs229.stanford.edu/proj2018/report/33.pdf).
- Authors' [reference implementation](https://github.com/jervisfm/cs229-project).
- Google [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
  licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html) for
  Apple silicon acceleration.
