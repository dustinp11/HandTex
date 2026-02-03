from flask import Flask, request, jsonify
from PIL import Image
import io
import sys
sys.path.append("..")
from models.vit import VitModel

app = Flask(__name__)
model = VitModel()


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    latex = model.predict(image)

    return jsonify({"latex": latex})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
