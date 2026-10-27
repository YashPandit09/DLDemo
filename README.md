# Handwritten Letter Recognizer (EMNIST Letters A-Z)

A web app where you draw a capital letter on a canvas and a CNN predicts it, plus the
notebook that trains and evaluates the model behind it. Built for a Deep Learning
mini-project; see `VIVA_NOTES.md` for the full technical writeup and viva prep.

**Workflow (module 1):** data -> model -> loss -> optimizer -> train -> evaluate -> deploy.
Every stage below maps to one of those steps.

**Deep-learning code size:** `data_utils.py` (86) + `models.py` (61) + `preprocess.py` (42)
+ notebook code cells (250) = **439 lines** -- within the project spec's 360-450 target.
Including `sanity_check.py` (51 lines) as well, as the course's own line-count rule does,
gives **490 lines**, under the course's 500-line cap. (`app.py`, `templates/index.html`,
and `download_data.py` are frontend/server/fetch code and are not counted either way.)

## Results summary

Trained on EMNIST Letters (124,800 train / 20,800 test, 28x28 grayscale, 26 balanced
classes). All numbers below are from the actual executed run in `letter_recognizer.ipynb`
and `outputs/results.json` -- nothing here is estimated.

| Model | Test accuracy | Test loss | Macro F1 | Params | Epochs run | Training time |
|---|---|---|---|---|---|---|
| MLP baseline | 89.78% | 0.327 | 0.898 | 237,210 | 8 (early stopped, restored epoch 5) | 23s |
| **CNN (final)** | **94.00%** | **0.190** | **0.940** | 424,090 | 15 (val_loss still improving at epoch 15) | 412s (~6.9 min) |

Both trained on CPU only (no GPU), full training split, `batch_size=128`.

**Experiments (validation accuracy, 40k-image subset, 4 epochs each -- see notebook for
figures and full conclusions):**

| Experiment | Result |
|---|---|
| E1: MLP vs CNN | MLP 86.03% vs CNN 90.54% -- CNN wins, confirmed by the final test numbers above |
| E2: optimizer | SGD 87.58% / RMSprop 90.07% / **Adam 90.54%** |
| E3: BatchNorm/Dropout ablation | full 90.54% / no_bn 91.13% / no_dropout 90.46% / neither 91.11% (no_dropout has the largest train-val gap, +2.4pp -- clearest overfitting signal) |
| E4: learning rate | 0.1 -> 37.0% / 0.01 -> 50.2% / **0.001 -> 90.5%** / 0.0001 -> 87.9% |
| E5: activation | relu 90.54% / leaky_relu 90.10% / elu 91.29% |
| E6: pooling | **max 90.54%** / avg 89.52% |

We re-ran the full notebook once after first writing it up (to add the MLP's macro-F1
to the results), and several of these experiment numbers shifted by 0.5-1.5 points
between runs despite the fixed `seed=42` -- CPU TensorFlow's multi-threaded ops aren't
bit-for-bit reproducible run to run. E3 and E5's rankings even flipped slightly (see the
notebook's conclusions for the honest, re-run-aware interpretation). The headline
CNN-vs-MLP and optimizer/learning-rate/pooling conclusions held up across both runs.

**Most confused letter pairs (test set):** L->I (196), I->L (147), G->Q (104), V->U (58),
Q->G (47) -- see the confusion matrix and misclassified-examples figures in
`outputs/figures/`.

**Preprocessing sanity check** (`sanity_check.py`, upscaled-EMNIST-as-drawing test,
n=1000): direct accuracy 93.90% vs pipeline (canvas) accuracy 91.00%, a 2.90-point gap
-- well within the 5-point pass threshold.

## Folder layout

```
letter-recognizer/
  README.md
  VIVA_NOTES.md
  requirements.txt
  download_data.py          # fetches the 4 dataset files if missing
  data_utils.py              # IDX loading, orientation fix, train/val/test split
  models.py                  # build_mlp(), build_cnn()
  preprocess.py               # canvas drawing -> 28x28 model input
  letter_recognizer.ipynb    # experiments, final training, evaluation (executed, outputs saved)
  sanity_check.py             # validates preprocess.py against the trained model
  app.py                      # Flask backend
  templates/index.html        # canvas frontend, no external CDN
  data/                        # 4 EMNIST Letters .gz files (not committed)
  outputs/
    emnist_cnn.keras
    history_*.json
    results.json
    figures/*.png
```

## Setup and run (Windows PowerShell)

Tested with Python 3.12, TensorFlow 2.20, Keras 3.12 on CPU only.

```powershell
cd letter-recognizer
pip install -r requirements.txt

# Dataset: the 4 emnist-letters-*.gz files should already be in data/.
# If missing, this tries to fetch them (see section below if it fails).
python download_data.py

# Run the full notebook top to bottom (re-trains everything, ~15-20 min on CPU).
# Outputs are already saved from the last run, so this step is optional.
python -m nbconvert --to notebook --execute --inplace letter_recognizer.ipynb

# Validate the canvas preprocessing pipeline against the trained model.
python sanity_check.py

# Start the web app, then open http://127.0.0.1:5000
python app.py
```

To just inspect the already-executed notebook (figures, tables, outputs) without
retraining, open it with `jupyter notebook letter_recognizer.ipynb` or view it directly
in VS Code / JupyterLab.

### If the dataset is missing

`download_data.py` tries the Hugging Face mirror
(`https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/<filename>`), then falls
back to the official NIST bundle (`https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip`,
~540 MB). If both fail, place these 4 files in `data/` manually:
`emnist-letters-train-images-idx3-ubyte.gz`, `emnist-letters-train-labels-idx1-ubyte.gz`,
`emnist-letters-test-images-idx3-ubyte.gz`, `emnist-letters-test-labels-idx1-ubyte.gz`.

## Known limitations

- **Merged case:** EMNIST Letters has one label per letter (26 classes), covering both
  upper and lower case. The model is fundamentally not case-sensitive -- it cannot tell
  a drawn "I" from a drawn "l", and several of the misclassifications we observed
  (e.g. a cursive lowercase "d" predicted as "a") come from genuinely ambiguous
  upper/lower-case shape overlap baked into the dataset itself, not a model defect.
- **Writer population:** EMNIST is drawn from a specific collection of writers (the
  original NIST handwriting sample forms). Accuracy on handwriting styles or
  conventions outside that population (e.g. different national schooling styles) is
  unverified -- we only measured accuracy on EMNIST's own test set and on our own
  upscaled-EMNIST and live-drawn sanity checks.
- **Canvas-vs-dataset gap:** real drawings differ from EMNIST's centering, scale and
  blur conventions. `sanity_check.py` shows a 2.9-point accuracy gap introduced by the
  preprocessing pipeline alone (using upscaled EMNIST images as a proxy for drawings);
  real freehand strokes may differ further.
- **Static image only:** no stroke order or pen dynamics are used (see module 6 below).
- **No data augmentation:** the model has only seen EMNIST's own 112,320 training
  images, so robustness to unfamiliar pen widths, slant, or drawing styles is limited.
- **Run-to-run noise:** repeating the exact same training (fixed `seed=42`, CPU only)
  produced test accuracy that varied by about half a point and some experiment
  rankings that changed slightly between two runs -- see "Results summary" above. We
  report numbers from one specific, fully-executed run throughout; a few points of
  difference on a repeat run would not be surprising.

## Ethics note

A wrong prediction here just mislabels a drawn letter -- low stakes in this toy UI, but
the same kind of model could plausibly sit behind a form-digitization or accessibility
tool, where a confident wrong answer matters more than a low-confidence one. The app
surfaces its **top-3** predictions with probabilities rather than a single answer, so a
user (or a human reviewer) can see when the model is unsure instead of trusting a silent
top-1 guess. No drawings or predictions are stored; each request is processed in memory
and discarded.

## Future work

- A sequence model (RNN/LSTM) over recorded pen strokes (x, y, time), not just the final
  rasterized image -- this project does not use an RNN (module 6 is out of scope here,
  see `VIVA_NOTES.md` for why) but it is the natural next step for online handwriting.
- Data augmentation (small rotations, shears, width jitter) to close the canvas-vs-EMNIST
  gap seen in `sanity_check.py`.
- Case-sensitive recognition, which would need a dataset with separate upper/lower-case
  labels (e.g. EMNIST ByClass/ByMerge) rather than EMNIST Letters.
