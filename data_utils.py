"""Loading and plotting helpers for the EMNIST Letters dataset."""
import gzip
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical

DATA_DIR = Path(os.environ.get("EMNIST_DIR", "data"))
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_images(path: Path) -> np.ndarray:
    """Read an IDX3 image file (as shipped, gzip-compressed) into (N, 28, 28) uint8."""
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


def load_labels(path: Path) -> np.ndarray:
    """Read an IDX1 label file (gzip-compressed) into a 1-D uint8 array."""
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_emnist_letters(data_dir: Path = DATA_DIR):
    """Load, fix orientation, normalize and split EMNIST Letters.

    Returns (x_train, y_train, x_val, y_val, x_test, y_test). Images are
    float32 in [0, 1] with shape (N, 28, 28, 1); labels are one-hot (N, 26).
    """
    x_train_raw = load_images(data_dir / "emnist-letters-train-images-idx3-ubyte.gz")
    y_train_raw = load_labels(data_dir / "emnist-letters-train-labels-idx1-ubyte.gz")
    x_test_raw = load_images(data_dir / "emnist-letters-test-images-idx3-ubyte.gz")
    y_test_raw = load_labels(data_dir / "emnist-letters-test-labels-idx1-ubyte.gz")

    assert x_train_raw.shape == (124800, 28, 28), x_train_raw.shape
    assert x_test_raw.shape == (20800, 28, 28), x_test_raw.shape

    # Module 5: EMNIST stores images transposed (rotated 90deg + mirrored).
    # Canvas drawings are upright, so training must be on upright images too.
    x_train_raw = x_train_raw.transpose(0, 2, 1)
    x_test_raw = x_test_raw.transpose(0, 2, 1)

    x_train_raw = x_train_raw.astype("float32") / 255.0
    x_test_raw = x_test_raw.astype("float32") / 255.0
    x_train_raw = x_train_raw.reshape(-1, 28, 28, 1)
    x_test_raw = x_test_raw.reshape(-1, 28, 28, 1)

    # Labels in the file are 1-26; shift to 0-25 for to_categorical / argmax.
    y_train_int = y_train_raw.astype("int64") - 1
    y_test_int = y_test_raw.astype("int64") - 1
    assert y_train_int.min() == 0 and y_train_int.max() == 25
    assert y_test_int.min() == 0 and y_test_int.max() == 25

    counts = np.bincount(y_train_int, minlength=26)
    print("Train class counts (A-Z):", dict(zip(CLASS_NAMES, counts.tolist())))

    y_train_cat = to_categorical(y_train_int, 26)
    y_test_cat = to_categorical(y_test_int, 26)

    # Module 1 workflow: 90/10 train/validation split, fixed seed for reproducibility.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_train_raw))
    n_val = int(0.1 * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train, y_train = x_train_raw[train_idx], y_train_cat[train_idx]
    x_val, y_val = x_train_raw[val_idx], y_train_cat[val_idx]

    return x_train, y_train, x_val, y_val, x_test_raw, y_test_cat


def plot_samples(x: np.ndarray, y: np.ndarray, n: int = 25) -> None:
    """Show an n-image grid with letter titles (y may be one-hot or integer)."""
    y_idx = np.argmax(y, axis=1) if y.ndim > 1 else y
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 1.3, rows * 1.3))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x[i].squeeze(), cmap="gray")
        plt.title(CLASS_NAMES[y_idx[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
