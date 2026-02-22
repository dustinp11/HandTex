from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "drawings")
os.makedirs(SAVE_DIR, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
    image.save(os.path.join(SAVE_DIR, filename))

    return jsonify({"latex": f"saved:{filename}"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
