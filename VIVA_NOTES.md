# VIVA_NOTES

Everything here is grounded in the actual run saved in `outputs/results.json` and
`letter_recognizer.ipynb`. No number below is estimated or invented.

---

## 1. The pitch

**30-second version:** We built a CNN that recognizes a hand-drawn capital letter (A-Z)
from a browser canvas. It's trained on EMNIST Letters (124,800 images), reaches 94.00%
accuracy on 20,800 held-out test images, and runs live behind a small Flask app that
preprocesses the drawing and returns the top-3 predicted letters.

**2-minute walkthrough (module 1 workflow):**

1. **Data** -- `data_utils.py` loads the 4 EMNIST IDX files, fixes a stored
   90-degree-rotation-plus-mirror transpose so letters are upright, normalizes pixels to
   `[0,1]`, shifts labels from 1-26 to 0-25, and splits 90/10 into train/validation
   (124,800 -> 112,320 train / 12,480 val). The 20,800-image test set stays untouched
   until the very end.
2. **Model** -- `models.py` defines an MLP baseline (237,210 params) and a CNN
   (424,090 params): two `Conv -> BatchNorm -> ReLU -> MaxPool` blocks, then
   `Dropout -> Dense(128) -> Dense(26, softmax)`.
3. **Loss** -- categorical cross-entropy against one-hot labels (`letter_recognizer.ipynb`
   section 2, `models.py` `.compile()` calls).
4. **Optimizer** -- Adam, lr=0.001, chosen in E2 (section 6) over SGD and RMSprop.
5. **Training** -- six controlled experiments (E1-E6, sections 5-10) on a 40k-image,
   4-epoch subset to cheaply compare design choices, then the chosen config trained on
   the full training split with `EarlyStopping(patience=3, restore_best_weights=True)`
   (section 11).
6. **Evaluation** -- test accuracy, per-class precision/recall/F1, a confusion matrix,
   most-confused pairs, and misclassified examples, computed exactly once on the test
   set (section 12).
7. **Deployment** -- `app.py` loads the saved `.keras` model once at startup;
   `preprocess.py` turns a canvas drawing into the same 28x28 format the model was
   trained on; `templates/index.html` is the drawing UI.

---

## 2. Syllabus mapping (with actual file/cell references)

| Module | Topics | Where it shows up |
|---|---|---|
| 1. Intro to DL | neuron, weights, bias, layers, MLP, workflow | `models.py: build_mlp()`; `letter_recognizer.ipynb` sections 0-2; this file section 1 |
| 2. Activations | ReLU, LeakyReLU, ELU, Softmax | `models.py: _activation_layer()`, both models' final `softmax`; E5, notebook section 9 |
| 3. Loss and optimizers | categorical cross-entropy, SGD, mini-batch, RMSprop, Adam, backprop, learning rate | `models.py .compile(loss="categorical_crossentropy")`; E2 (section 6), E4 (section 8); this file section 4 |
| 4. Essential concepts | hidden features, init, BatchNorm, Dropout, early stopping, P/R/F1, hyperparameter tuning | `models.py: build_cnn()`; E3 (section 7); `letter_recognizer.ipynb` section 11-12 (EarlyStopping, `classification_report`) |
| 5. CNN | filters, padding, stride, pooling, FC layers, why MLPs fail on images | `models.py: build_cnn()`; E1 (section 5), E6 (section 10); section 13 (filters/feature maps figures) |
| 6. RNN | sequential data | **Not used** -- see section 7 of this file ("what we would improve") and `README.md` future work |
| Deployment/ethics | deployment, ethical considerations | `app.py`, `templates/index.html`; `README.md` "Known limitations" and "Ethics note" |

---

## 3. File-by-file / cell-by-cell explanation

### `data_utils.py`
- `load_images` / `load_labels`: read the raw IDX format directly with `np.frombuffer`,
  skipping the 16-byte (images) / 8-byte (labels) header -- no external IDX library
  needed, and it's transparent about exactly what bytes mean what.
- `load_emnist_letters`: asserts the raw shapes `(124800,28,28)` / `(20800,28,28)`
  immediately after loading, so a corrupt or wrong file fails loudly instead of
  training on garbage. Applies `.transpose(0,2,1)` -- **this is the single most
  important line in the whole project**: EMNIST stores each image rotated 90 degrees
  and mirrored, and if this were skipped, the model would train on sideways/mirrored
  letters while the deployed canvas always sends upright ones, silently breaking the
  whole app. Labels are shifted `-1` because the files use 1-26, not 0-25.
  `np.random.default_rng(42)` gives a reproducible 90/10 train/val split.
- `plot_samples`: shows a titled image grid; called immediately after loading
  (`letter_recognizer.ipynb` section 1) specifically to catch an orientation bug
  before any time is spent training.

### `models.py`
- `build_mlp`: `Flatten -> Dense(256,relu) -> Dense(128,relu) -> Dense(26,softmax)`,
  237,210 params (verified against `model.summary()`, see section 4 below).
- `_activation_layer`: a small helper because LeakyReLU/ELU need to be added as
  *layers* in Keras, not as the `activation=` string Dense/Conv2D accept directly.
- `build_cnn`: each block is **Conv -> BatchNorm -> activation -> Pool**, matching the
  "normalize before nonlinearity" convention discussed in module 4. `use_bn`,
  `dropout`, `pool`, `activation`, `padding` are all parameters so the same function
  builds every E1-E6 variant without copy-pasting the architecture six times.
  Default config: 424,090 params.

### `preprocess.py` -- `canvas_to_model_input`
1. Convert to grayscale, threshold at 20 to build a "what's actually drawn" mask;
   fewer than 30 lit pixels is treated as blank (returns `None`).
2. Crop to the bounding box of drawn pixels.
3. Pad to a centered square with a 20%-of-side margin -- without this, a tall narrow
   letter like "I" would get horizontally stretched when forced into a square.
4. Gaussian blur (radius 1) -- EMNIST's own images are blurred before their original
   downsampling to 28x28, so this makes a sharp canvas stroke look statistically more
   like what the model was trained on.
5. Resize to 28x28 with LANCZOS (antialiasing) resampling.
6. Return both the normalized `(1,28,28,1)` float32 model input and the raw uint8
   28x28 preview (for the "what the model sees" UI panel).

### `sanity_check.py`
The single most important pre-deployment check for this kind of project: it does not
trust that `preprocess.py` works just because it runs without errors. It takes 1,000
real test images, upscales each to 280x280 on a black canvas (imitating what a drawing
looks like), runs them through the full canvas pipeline, and compares accuracy against
feeding the same images directly to the model. Result: **93.90% direct vs 91.00%
pipeline, a 2.90-point gap** -- comfortably inside the 5-point pass bar, so the
preprocessing is not silently destroying information.

### `app.py`
- Loads `outputs/emnist_cnn.keras` once at import time (not per-request) -- loading a
  Keras model is slow; doing it once keeps `/predict` responsive.
- `/predict` decodes the incoming base64 PNG, runs it through `canvas_to_model_input`,
  and either returns `{"empty": true, ...}` (blank canvas) or the top-3 letters with
  probabilities plus a scaled-up preview of exactly what the model received.

### `templates/index.html`
Plain HTML/CSS/JS, no build step or CDN (must work offline). The canvas uses
`lineCap/lineJoin = "round"` plus a filled circle on `pointerdown` so single clicks and
stroke joins look like continuous pen strokes rather than straight-line segments.
`mouseup` is bound on `window`, not the canvas, so a stroke that ends outside the
canvas boundary still ends the draw instead of leaving `drawing = true` stuck. A 400ms
debounce after the pen lifts triggers auto-predict without spamming `/predict` on every
pixel of movement.

### `letter_recognizer.ipynb` (49 cells, executed, outputs saved)
Sections 0-4 set up config/data/models and a shared `run_configs` /
`report_and_plot` helper (one small loop instead of six copy-pasted training blocks).
Sections 5-10 are E1-E6, each: train -> print a results table -> plot overlaid
val-accuracy/val-loss curves -> markdown conclusion from the real numbers. Section 11
trains the final CNN on full data with early stopping; section 12 evaluates on the
untouched test set exactly once (accuracy/loss, classification report, confusion
matrix, most-confused pairs, misclassified grid) and also trains a final MLP the same
way, computing its own `classification_report` and macro F1, so E1's conclusion is
backed by real test numbers for both models. Section 13 visualizes the first conv
layer's 32 filters and its 32 feature maps for one test letter. Section 14 saves the
model and `results.json`. Section 15 is the final summary.

---

## 4. Formulas, tied to this project's actual numbers

### Conv output size: `floor((n + 2p - f) / s) + 1`

Both conv layers use `padding="same"`, `kernel=3`, `stride=1`, which Keras implements
as `p=1`:

- Conv1: `floor((28 + 2*1 - 3)/1) + 1 = floor(27) + 1 = 28` -> stays 28x28.
  MaxPool(2,2): `floor((28-2)/2)+1 = 14` -> 14x14.
- Conv2 (input 14x14): `floor((14 + 2*1 - 3)/1) + 1 = 14` -> stays 14x14.
  MaxPool(2,2): `floor((14-2)/2)+1 = 7` -> 7x7.
- Flatten: `7 * 7 * 64 = 3136`, matching `Dense(128)`'s input size exactly.

This is why the architecture comment lists `28x28x32 -> 14x14x32 -> 14x14x64 -> 7x7x64
-> 3136 -> 128 -> 26`.

### Parameter counts: `(k*k*C_in + 1) * C_out` (conv), `(n_in + 1) * n_out` (dense)

Every layer below is reproduced by hand and matched against `model.summary()`:

| Layer | Formula | Computed | `model.summary()` |
|---|---|---|---|
| Conv2D(32, 3x3, in=1) | `(3*3*1+1)*32` | 320 | 320 |
| BatchNorm(32) | `4*32` (gamma, beta, mov.mean, mov.var; 2*32 trainable + 2*32 non-trainable) | 128 | 128 |
| Conv2D(64, 3x3, in=32) | `(3*3*32+1)*64` | 18,496 | 18,496 |
| BatchNorm(64) | `4*64` | 256 | 256 |
| Dense(128, in=3136) | `(3136+1)*128` | 401,536 | 401,536 |
| Dense(26, in=128) | `(128+1)*26` | 3,354 | 3,354 |
| **CNN total** | sum | **424,090** (423,898 trainable + 192 non-trainable) | 424,090 |
| Dense(256, in=784) | `(784+1)*256` | 200,960 | 200,960 |
| Dense(128, in=256) | `(256+1)*128` | 32,896 | 32,896 |
| Dense(26, in=128) | `(128+1)*26` | 3,354 | 3,354 |
| **MLP total** | sum | **237,210** | 237,210 |

### Softmax and categorical cross-entropy, and `dL/dz = p - y`

Softmax: `p_i = exp(z_i) / sum_k exp(z_k)`. Cross-entropy against one-hot `y`:
`L = -sum_i y_i * log(p_i)`.

Derivation of the gradient w.r.t. the pre-softmax logits `z`:

```
dL/dp_k = -y_k / p_k
dp_k/dz_i = p_k * (delta_ik - p_i)        (softmax Jacobian)

dL/dz_i = sum_k (dL/dp_k) * (dp_k/dz_i)
        = sum_k [-y_k / p_k] * [p_k * (delta_ik - p_i)]
        = sum_k [-y_k * (delta_ik - p_i)]
        = -y_i + p_i * sum_k y_k
        = p_i - y_i                        (since y is one-hot, sum_k y_k = 1)
```

This clean result (`p - y`) is exactly why cross-entropy + softmax is the standard
classification pairing -- the gradient flowing into the last layer is simply
"predicted probability minus the true one-hot label."

### Backpropagation by the chain rule, one dense layer

For a layer `z = W a_prev + b`, `a = f(z)`, given `dL/da` from the layer above:

```
dL/dz      = dL/da (elementwise) * f'(z)
dL/dW      = dL/dz  (outer)  a_prev^T          (averaged over the mini-batch)
dL/db      = mean over batch of dL/dz
dL/da_prev = W^T * dL/dz                        (passed to the previous layer)
```

### SGD, RMSprop, Adam, and mini-batch SGD

Let `g` be the gradient of the loss w.r.t. a weight, `lr` the learning rate:

- **SGD:** `w <- w - lr * g` (E2 used lr=0.01, no momentum).
- **RMSprop:** `s <- rho*s + (1-rho)*g^2; w <- w - lr * g / (sqrt(s) + eps)`
  (`rho=0.9` default; E2 used lr=0.001).
- **Adam:** `m <- b1*m + (1-b1)*g; v <- b2*v + (1-b2)*g^2;`
  `m_hat = m/(1-b1^t); v_hat = v/(1-b2^t); w <- w - lr * m_hat/(sqrt(v_hat)+eps)`
  (`b1=0.9, b2=0.999` defaults; used everywhere as the final optimizer, lr=0.001).
- **Mini-batch SGD:** with `BATCH_SIZE=128` and 112,320 training images, each epoch is
  `112320/128 = 878` weight-update steps, each averaging the gradient over 128
  examples -- a middle ground between full-batch GD (one noisy-free but slow step per
  epoch) and pure SGD (one step per example, very noisy).

### BatchNorm forward pass

`y = gamma * (x - mu_B) / sqrt(var_B + eps) + beta`, where `mu_B`, `var_B` are the
mean/variance over the current mini-batch (per channel). `gamma`, `beta` are learned;
`eps` (Keras default `1e-3`) avoids division by zero. At inference, `mu_B`/`var_B` are
replaced by a running (`moving_mean`/`moving_variance`) estimate accumulated during
training -- which is exactly the 2 non-trainable params per BatchNorm layer seen in
`model.summary()`.

### Dropout (inverted dropout)

Training: each unit is zeroed with probability `rate` (here `0.4`), and the surviving
activations are scaled by `1/(1-rate)` so the expected activation magnitude matches
inference. Inference: dropout is a no-op (identity) -- all units active, no scaling.
This is also why E3's `no_dropout` run showed a **positive** train-val accuracy gap
while the dropout-enabled runs showed a slightly **negative** one (section 7): Keras
reports training accuracy computed *with* dropout active (so it looks artificially
lower), while validation accuracy is always computed in inference mode.

### Early stopping

`EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`: after each
epoch, if `val_loss` hasn't improved for 3 consecutive epochs, training stops and the
weights are rolled back to whichever epoch had the lowest `val_loss` seen so far.

**What actually happened in our run:** the final CNN's `val_loss` was still falling at
the last epoch -- its minimum over all 15 epochs was **epoch 15 itself (0.17833)**, with
epoch 13 (0.18059) the next best. Patience never had a chance to trigger a stop (it
needs 3 *consecutive* non-improving epochs), so training simply reached
`FINAL_MAX_EPOCHS=15`, and `restore_best_weights=True` restored those same final
weights -- a no-op restoration this time, though the mechanism is identical to a run
where it would matter. (This also suggests `FINAL_MAX_EPOCHS=15` may have been slightly
conservative here; a higher cap might have improved further, though the last few
epochs' gains were small.) The MLP baseline *did* trigger a real stop-and-restore: its
`val_loss` bottomed out at epoch 5 (0.33183), then got worse for 3 straight epochs
(6, 7, 8 -- 0.33599, 0.33942, 0.34222), so training stopped after epoch 8 and epoch 5's
weights were restored.

### Precision, recall, F1 -- worked example (class "I")

From the test-set classification report: support=800, precision=0.7430,
recall=0.7950, giving TP=636, FN=164, FP=220 (solved from `recall=TP/support` and
`precision=TP/(TP+FP)`):

```
Precision = TP / (TP + FP) = 636 / (636 + 220) = 0.7430
Recall    = TP / (TP + FN) = 636 / (636 + 164) = 0.7950
F1        = 2 * P * R / (P + R) = 2*0.7430*0.7950 / (0.7430+0.7950) = 0.7681
```

**Macro average** (unweighted mean of all 26 classes' precision/recall/F1, treats
every letter equally regardless of how many test examples it has):
precision=0.9408, recall=0.9400, F1=0.9399 -- all close to the overall accuracy
(0.9400) because EMNIST Letters is perfectly class-balanced (800 test images/letter),
so macro and accuracy rarely diverge much here. "I" (F1=0.7681) and "L" (F1=0.7529)
are the two weakest classes -- exactly the pair most confused with each other. The MLP
baseline's own classification report shows the same pattern more sharply: its macro F1
is 0.8978 versus the CNN's 0.9399, and its "I" class (F1=0.6713, recall only 0.6038) is
noticeably worse than the CNN's -- the MLP struggles hardest on exactly the class that
is already hardest for the better model.

---

## 5. Experiment results table (with interpretation)

All validation accuracies below are from the 40k-image, 4-epoch subset unless noted.
**A note on reproducibility:** we re-ran this notebook twice with the same `seed=42` on
the same CPU (once to add the MLP's own classification report), and several of these
numbers shifted by 0.5-1.5 points between runs, with E3 and E5's rankings changing
slightly -- `tf.keras.utils.set_random_seed(42)` fixes initialization and shuffling,
but not the exact floating-point order of CPU multi-threaded reductions. The table
below is from the final, fully-executed run; see section 7 for what this means for how
hard we lean on close results.

| Exp | Result | Interpretation |
|---|---|---|
| **E1** MLP vs CNN | MLP 86.03% / CNN 90.54% (subset); **MLP 89.78% / CNN 94.00% (full test set)** | The CNN wins by over 4 points on the real test set despite having fewer than 2x the MLP's parameters. Convolution's weight-sharing across spatial positions is doing real work, not just adding capacity. |
| **E2** Optimizer | SGD 87.58% / RMSprop 90.07% / **Adam 90.54%** | Adaptive per-weight step sizes (RMSprop, Adam) converge faster than plain SGD within a fixed small epoch budget; Adam's added momentum term edges out RMSprop. |
| **E3** BN/Dropout ablation | full 90.54% / no_bn 91.13% / no_dropout 90.46% / neither 91.11% | `no_dropout` has by far the largest train-val accuracy gap (+2.4pp vs at most +1.2pp elsewhere) -- the clearest overfitting signal, confirming Dropout's regularizing effect. Validation accuracy itself is noisier: the two BatchNorm-free configs actually scored *higher* here, suggesting BN's own scale/shift parameters hadn't fully settled in just 4 epochs -- its benefit shows up in the smooth, non-divergent full 15-epoch curve instead. |
| **E4** Learning rate | 0.1->37.0% / 0.01->50.2% / **0.001->90.5%** / 0.0001->87.9% | Classic U-shape: both larger rates overshoot and fail to converge in 4 epochs, 0.0001 is too slow to catch up, 0.001 is the sweet spot for this model/optimizer/epoch-budget combination. |
| **E5** Activation | relu 90.54% / leaky_relu 90.10% / **elu 91.29%** | All three are within 1.2 points of each other; with BatchNorm already stabilizing activations, none shows a decisive, repeatable edge at this short budget. ReLU stays the default for simplicity. |
| **E6** Pooling | **max 90.54%** / avg 89.52% | Max pooling preserves a stroke's strongest signal in each window; average pooling blends it with surrounding black background, losing some edge sharpness. |

---

## 6. Likely viva questions and concise answers

1. **Why a CNN instead of an MLP for images?** Convolution shares the same small
   filter across every spatial position, so a learned stroke/edge detector
   generalizes everywhere in the image; an MLP must relearn each pattern separately
   per pixel position after flattening. Confirmed here: CNN 94.00% vs MLP 89.78% test
   accuracy (E1).
2. **Why `padding="same"`?** It keeps the spatial size unchanged after each conv
   (28->28, 14->14), so the only size reduction comes from the explicit 2x2 pooling
   layers -- makes the shape arithmetic easy to reason about and matches the
   `28->14->7` pipeline we designed around.
3. **What does BatchNorm do, and why before the activation?** It re-centers and
   re-scales each mini-batch's pre-activation values (`(x-mu)/sqrt(var+eps)`, then
   learned `gamma`/`beta`), keeping the input distribution to the activation function
   stable across training; placing it before the activation means the nonlinearity
   always sees a consistently-scaled input instead of one that drifts as earlier
   layers' weights change.
4. **Why dropout 0.4?** It's the architecture's spec default, chosen because the
   `Dense(128)` layer right after a 3,136-unit flatten is the part of the network most
   prone to overfitting; E3 confirms it: `no_dropout` has by far the largest train-val
   accuracy gap (+2.4pp) of the four configs tested, the clearest overfitting signal
   in that experiment.
5. **Why Adam?** E2 shows it reaches the highest validation accuracy (90.54%) of the
   three optimizers tried, and it's the de facto default for this kind of problem.
6. **What happens with lr=0.1?** Each update overshoots the loss surface badly;
   validation accuracy stalls at 37.0% after 4 epochs (E4) -- the model barely learns
   anything useful at that step size.
7. **Why early stopping with `restore_best_weights=True`?** It protects against the
   model's weights drifting to a slightly worse point after its best epoch. In our
   final run `val_loss` was actually still falling at epoch 15 (the last epoch), so
   restoration was a no-op here -- but the mechanism matters in general, and it did
   trigger a real rollback for the MLP baseline (restored epoch 5 after 3 worse
   epochs).
8. **Why isn't accuracy alone enough?** It hides *which* classes struggle. Here,
   overall accuracy is 94.0%, but "I" and "L" individually sit around 75-77% F1 --
   averaging over 26 balanced classes can mask that two letters are doing much worse
   than the rest.
9. **Why were labels shifted by 1?** The EMNIST label files use 1-26 (NIST/MATLAB
   convention); Keras's `to_categorical` and `argmax`-based class indexing expect
   0-indexed labels, so `-1` is required before one-hot encoding.
10. **Why the transpose?** EMNIST images are stored rotated 90 degrees and mirrored
    relative to how they'd naturally be drawn; without `.transpose(0,2,1)`, the model
    would train on sideways/mirrored letters while canvas drawings are always upright,
    making the deployed model nearly useless despite good training metrics.
11. **Why categorical cross-entropy, not MSE?** Cross-entropy is the loss derived from
    maximum-likelihood estimation for a categorical (softmax) output and gives the
    clean `p-y` gradient at the logits; MSE on probabilities produces much smaller,
    vanishing gradients when predictions are confidently wrong, slowing learning.
12. **What are vanishing gradients, and how do ReLU/BatchNorm help?** Deep networks
    with saturating activations (sigmoid/tanh) can produce gradients that shrink to
    near-zero through many layers, stalling learning in early layers. ReLU's gradient
    is either 0 or 1 (no squashing for positive inputs), and BatchNorm keeps
    activations in a well-scaled range, both of which keep gradients from shrinking
    layer over layer.
13. **What does a learned filter look like / learn?** `outputs/figures/filters.png`
    shows the 32 first-layer 3x3 filters as small grayscale patches -- mostly
    edge/stroke-orientation detectors rather than anything a human would directly
    recognize as a shape.
14. **Why mini-batches instead of single examples or the full dataset?** Full-batch
    gradient descent is slow (one update per full pass) and noise-free, which can get
    stuck in sharp minima; single-example SGD is fast per step but very noisy.
    `batch_size=128` is a practical middle ground, giving 878 updates/epoch on our
    112,320-image training set.
15. **Why a validation split, separate from the test set?** The validation set drives
    every decision made *during* development -- comparing E1-E6, choosing when to
    early-stop -- so if it were also the final reporting number, those decisions would
    be implicitly fit to it, inflating the real-world estimate.
16. **Why is the test set used only once?** Using it repeatedly to guide decisions
    would turn it into a second validation set, and the reported accuracy would no
    longer be an honest estimate of performance on genuinely unseen data.
17. **How is the canvas image preprocessed, and why each step?** Grayscale ->
    threshold/bounding-box crop -> pad to a centered square with a 20% margin -> slight
    Gaussian blur -> resize to 28x28 with LANCZOS. Each step mirrors something true of
    EMNIST's own images (centered, slightly blurred, consistent scale), minimizing the
    gap between training and deployment input distributions.
18. **What would break if the orientation fix were missing?** The model would train
    to high accuracy on *rotated/mirrored* letters, but every real canvas drawing
    (always upright) would be fed in the "wrong" orientation relative to training --
    predictions would be close to random.
19. **Why do I/L get confused?** EMNIST Letters merges upper and lower case into one
    label; a handwritten lowercase "l" and uppercase "I" can be visually identical
    single vertical strokes, with literally no distinguishing shape feature. This is
    the single largest confusion pair in our results (196 + 147 = 343 misclassifications).
20. **Why do Q/G get confused?** Both have a loop with a small tail/stroke breaking its
    circle, and handwriting styles vary enough in where that tail sits that the shapes
    overlap substantially (104 + 47 = 151 misclassifications).
21. **Why no RNN here?** EMNIST Letters provides only the final rasterized image per
    letter, not the sequence of pen strokes that produced it -- there's no temporal/
    sequential structure in a single static 28x28 image for a recurrent layer to
    exploit. An RNN would make sense for *online* handwriting data (stroke
    coordinates over time), which is out of scope here (see `README.md` future work).
22. **What is weight initialization here, and why does it matter?** Left at Keras's
    default, Glorot/Xavier uniform, which samples initial weights from a range scaled
    by the layer's fan-in/fan-out so activations and gradients start with comparable
    variance layer to layer -- a bad (e.g. all-zero or too-large) initialization can
    prevent learning entirely or cause early instability.
23. **Why does `no_dropout` show the biggest train-val accuracy gap of the four E3
    configs?** Keras computes training accuracy with dropout active when dropout is
    enabled (randomly zeroing units), which depresses the reported training number
    relative to validation (always evaluated in inference mode) for the
    dropout-enabled configs -- so `no_dropout`'s gap being +2.4pp, more than double any
    other config's, is a genuine overfitting signal rather than just this metric
    quirk working in the other direction.
24. **How many parameters does the CNN have, and where do they come from?** 424,090
    total (423,898 trainable + 192 non-trainable in BatchNorm's running stats) -- see
    the full per-layer breakdown in section 4 above; the two Dense layers
    (401,536 + 3,354) account for over 95% of all trainable parameters, not the conv
    layers.
25. **Why does the Dense(128) layer dominate the parameter count?** It connects every
    one of the 3,136 flattened conv-output units to 128 units
    (`(3136+1)*128 = 401,536`), while a conv layer's parameters are shared across all
    spatial positions and only scale with filter size and channel count, not image
    size -- this asymmetry is exactly why convolution is so much more
    parameter-efficient per unit of spatial coverage.
26. **What's the difference between the 20,800 test images and the 12,480 validation
    images?** Both come from data the model never trains on directly, but validation
    was used to make modelling decisions (experiment comparisons, early stopping);
    test was reserved untouched specifically to give one honest, decision-free
    accuracy number at the end.
27. **Why use `classification_report`'s macro average instead of just accuracy?**
    Macro average treats every class equally regardless of frequency, surfacing
    weak classes (I: 0.77 F1, L: 0.75 F1) that a single accuracy number (94.0%)
    would hide; here it happens to be very close to accuracy because EMNIST Letters
    is perfectly class-balanced. The MLP's own report shows the same classes (I, L)
    as its weakest too, but noticeably worse (I: F1 0.67), and its overall macro F1
    (0.898) is a full 4.2 points below the CNN's (0.940) -- consistent with the
    accuracy gap.
28. **Why LANCZOS resizing in preprocessing rather than nearest-neighbor?** LANCZOS
    is an antialiasing filter -- it averages nearby pixels when downscaling, avoiding
    the jagged/aliased edges nearest-neighbor would introduce, which better matches
    how EMNIST's own images were originally downsampled.
29. **What does `sanity_check.py` actually prove, and what are its limits?** It
    proves the preprocessing pipeline doesn't badly damage prediction accuracy
    compared to feeding model-format images directly (93.90% vs 91.00%, a 2.9-point
    gap). It does *not* prove the model works well on genuine freehand strokes, since
    it uses upscaled EMNIST images (already clean, centered, single-stroke-width) as
    a proxy for a real drawing -- real human strokes vary more in width, slant, and
    noise.
30. **Why is the final MLP's early stopping behavior different from the CNN's?**
    They're independent training runs with different architectures and convergence
    speeds; the MLP's `val_loss` bottoms out early (epoch 5) and then worsens for 3
    straight epochs (triggering patience=3 after epoch 8, so epoch 5's weights are
    restored), while the CNN's `val_loss` was still falling at epoch 15 and never
    accumulated 3 bad epochs in a row before hitting `FINAL_MAX_EPOCHS=15`.
31. **How would you make this case-sensitive?** EMNIST Letters fundamentally can't
    support that (one label per letter); it would need a dataset like EMNIST ByClass
    or ByMerge with separate upper/lower-case labels, plus likely a larger/deeper
    model given the added class count and the increased ambiguity between some
    upper/lower pairs.
32. **What's the biggest limitation of this project, honestly?** The gap between
    EMNIST's writer population and arbitrary real-world handwriting -- we only
    validated accuracy on EMNIST's own test set and on EMNIST-derived "simulated
    drawings," not on a broad sample of real human strokes from diverse writers.
33. **Is `tf.keras.utils.set_random_seed(42)` enough for reproducible results?** Not
    fully, on CPU. We re-ran the identical notebook twice and got CNN test accuracy of
    93.50% the first time and 94.00% the second, with some E3/E5 experiment rankings
    flipping between runs. The seed fixes weight initialization and data shuffling,
    but TensorFlow's CPU ops use multi-threaded floating-point reductions whose
    summation order isn't guaranteed identical run to run, so bit-exact
    reproducibility would need either single-threaded execution or
    `TF_DETERMINISTIC_OPS`-style settings we did not apply here.

---

## 7. What we would improve, and honest limitations

- The canvas-to-model gap (2.9 points in `sanity_check.py`) could likely shrink
  further by tuning the blur radius/margin against a larger set of *real* drawn
  letters rather than only upscaled EMNIST images.
- E3 suggests BatchNorm's benefit is underexplored at only 4 experiment epochs; a
  longer-epoch ablation might show a clearer accuracy or convergence-speed gap.
- No data augmentation was used (out of scope per the project brief), so the model
  has only ever seen EMNIST's own stroke widths and styles -- augmentation
  (rotation, shear, width jitter) is the most likely lever to improve robustness to
  real handwriting.
- I/L and Q/G remain hard-confused by construction (merged case, overlapping shapes)
  -- no amount of training fixes an ambiguity that exists in the label scheme itself.
- **Run-to-run noise is real and larger than we initially assumed.** We trained and
  evaluated on one CPU with no repeated-seed averaging; re-running the identical
  notebook (same `seed=42`) a second time moved the final CNN's test accuracy by half
  a point (93.50% -> 94.00%) and flipped which E3/E5 configuration scored highest.
  Close experiment results (within roughly 1-1.5pp) should be read as "statistically
  indistinguishable at this sample size," not as a confident ranking -- a proper
  comparison would average several seeds per configuration, which we did not have
  time budget for here.
