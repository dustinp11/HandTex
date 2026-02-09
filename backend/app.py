from flask import Flask, request, jsonify
from PIL import Image
import io
import sys
import os, time, uuid

sys.path.append("..")
#from models.vit import VitModel

app = Flask(__name__)
#model = VitModel()


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]

    # Make uploads folder
    os.makedirs("uploads", exist_ok=True)

    # Save raw upload as PNG
    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    save_path = os.path.join("uploads", filename)
    file.save(save_path)

    # Open it to validate it's a real image
    img = Image.open(save_path)
    width, height = img.size
    mode = img.mode
    file_size = os.path.getsize(save_path)

    print(f"[UPLOAD OK] saved={save_path} size={file_size} bytes dim={width}x{height} mode={mode}")
    # image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # latex = model.predict(image)

    # return jsonify({"latex": latex})

    return jsonify({
        "ok": True,
        "saved_as": save_path,
        "bytes": file_size,
        "width": width,
        "height": height,
        "mode": mode
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
