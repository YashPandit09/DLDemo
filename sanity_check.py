"""Acceptance test for preprocess.py: compare accuracy via the canvas pipeline vs direct input."""
import numpy as np
from PIL import Image
from tensorflow import keras

from data_utils import load_emnist_letters
from preprocess import canvas_to_model_input

N_SAMPLES = 1000
MODEL_PATH = "outputs/emnist_cnn.keras"


def upscale_to_canvas(img28: np.ndarray) -> Image.Image:
    """Imitate a hand drawing: paste a 28x28 EMNIST letter upscaled onto a black 280x280 canvas."""
    pil = Image.fromarray((img28 * 255).astype(np.uint8)).resize((280, 280), Image.LANCZOS)
    canvas = Image.new("L", (280, 280), 0)
    canvas.paste(pil, (0, 0))
    return canvas


def main():
    _, _, _, _, x_test, y_test = load_emnist_letters()
    y_true_all = np.argmax(y_test, axis=1)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_test), size=N_SAMPLES, replace=False)
    model = keras.models.load_model(MODEL_PATH)

    pipeline_inputs, pipeline_true, skipped = [], [], 0
    for i in idx:
        result = canvas_to_model_input(upscale_to_canvas(x_test[i, :, :, 0]))
        if result is None:
            skipped += 1
            continue
        model_input, _ = result
        pipeline_inputs.append(model_input[0])
        pipeline_true.append(y_true_all[i])

    direct_pred = np.argmax(model.predict(x_test[idx], verbose=0), axis=1)
    pipeline_pred = np.argmax(model.predict(np.array(pipeline_inputs), verbose=0), axis=1)

    direct_acc = (direct_pred == y_true_all[idx]).mean()
    pipeline_acc = (pipeline_pred == np.array(pipeline_true)).mean()
    gap = (direct_acc - pipeline_acc) * 100

    print(f"Direct accuracy:   {direct_acc:.4f} (n={len(idx)})")
    print(f"Pipeline accuracy: {pipeline_acc:.4f} (n={len(pipeline_true)}, skipped={skipped})")
    print(f"Gap: {gap:.2f} percentage points -> {'PASS' if abs(gap) <= 5 else 'FAIL'}")


if __name__ == "__main__":
    main()
