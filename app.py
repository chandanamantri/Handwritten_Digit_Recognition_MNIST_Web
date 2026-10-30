from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageOps, ImageFilter

app = Flask(__name__)
MODEL_PATH = Path("digit_model.joblib")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def preprocess_image(image):
    image = image.convert("L")
    arr = np.asarray(image, dtype=np.uint8)

    # Browser canvas: white background, black ink.
    mask = arr < 220
    if not mask.any():
        raise ValueError("Please draw one digit before predicting.")

    ys, xs = np.where(mask)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1

    cropped = Image.fromarray(arr[top:bottom, left:right], mode="L")
    cropped = ImageOps.invert(cropped)

    # MNIST-style: put digit in a square with generous padding.
    w, h = cropped.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(cropped, ((side - w) // 2, (side - h) // 2))

    # 28x28 MNIST-sized image.
    small = canvas.resize((20, 20), Image.Resampling.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(0.35))

    mnist = Image.new("L", (28, 28), 0)
    mnist.paste(small, (4, 4))

    # MNIST-style intensity: white digit on black background, 0..255.
    return np.asarray(mnist, dtype=np.float32).reshape(1, -1) / 255.0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model

    if model is None:
        return jsonify({"success": False,
                        "error": "Model not found. Run: python train_model.py"}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image received."}), 400

    try:
        image = Image.open(request.files["image"].stream)
        features = preprocess_image(image)

        probabilities = model.predict_proba(features)[0]
        order = np.argsort(probabilities)[::-1]
        prediction = int(order[0])

        alternatives = [
            {"digit": int(d), "confidence": round(float(probabilities[d] * 100), 2)}
            for d in order[:3]
        ]

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(float(probabilities[prediction] * 100), 2),
            "alternatives": alternatives
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=False)
