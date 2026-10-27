"""Convert a canvas drawing (PNG bytes or PIL Image) into the model's 28x28 input."""
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

THRESHOLD = 20          # pixel value above which a pixel counts as "drawn"
MIN_PIXELS = 30         # fewer lit pixels than this -> treat the canvas as empty
MARGIN_FRACTION = 0.2   # margin added on each side, as a fraction of the square's side
BLUR_RADIUS = 1         # EMNIST originals were blurred before downsampling; match that


def canvas_to_model_input(image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (model_input (1,28,28,1) float32, preview uint8 28x28), or None if blank."""
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    gray = np.array(image.convert("L"))

    mask = gray > THRESHOLD
    if mask.sum() < MIN_PIXELS:
        return None

    rows, cols = np.where(mask)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    crop = gray[top:bottom + 1, left:right + 1]

    # Pad the crop to a centered square with a margin, so resizing to 28x28
    # does not stretch the letter (EMNIST images are centered this way too).
    h, w = crop.shape
    side = max(h, w)
    margin = int(side * MARGIN_FRACTION)
    canvas_side = side + 2 * margin
    square = np.zeros((canvas_side, canvas_side), dtype=np.uint8)
    y_off, x_off = (canvas_side - h) // 2, (canvas_side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop

    square_img = Image.fromarray(square).filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    preview = np.array(square_img.resize((28, 28), Image.LANCZOS), dtype=np.uint8)

    model_input = (preview.astype("float32") / 255.0).reshape(1, 28, 28, 1)
    return model_input, preview
