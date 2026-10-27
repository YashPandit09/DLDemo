"""Flask backend: loads the trained CNN once and serves predictions for canvas drawings."""
import base64
import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow import keras

from data_utils import CLASS_NAMES
from preprocess import canvas_to_model_input

app = Flask(__name__)
model = keras.models.load_model("outputs/emnist_cnn.keras")


def data_url_to_image(data_url: str) -> Image.Image:
    """Decode a `data:image/png;base64,...` string into a PIL image."""
    _, encoded = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def image_to_data_url(img28: np.ndarray) -> str:
    """Encode the 28x28 'what the model sees' preview as a scaled-up PNG data URL."""
    buf = io.BytesIO()
    Image.fromarray(img28).resize((140, 140), Image.NEAREST).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.get_json(force=True).get("image", "")
    image = data_url_to_image(data_url)
    result = canvas_to_model_input(image)
    if result is None:
        return jsonify({"empty": True, "message": "Draw a letter first."})

    model_input, preview = result
    probs = model.predict(model_input, verbose=0)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"letter": CLASS_NAMES[i], "probability": float(probs[i])} for i in top3_idx]

    return jsonify({"empty": False, "top3": top3, "preview": image_to_data_url(preview)})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
