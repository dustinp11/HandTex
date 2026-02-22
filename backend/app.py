from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, os
from datetime import datetime

import numpy as np
import torch
from pickle import load as pkl_load

from models.vit_lora_lstm_attn import ViTLatexModelLoRA

app = Flask(__name__)
CORS(app)

# Where to save incoming drawings (same as your existing code)
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "drawings")
os.makedirs(SAVE_DIR, exist_ok=True)

# Where your checkpoint/tokenizer live
ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load artifacts once ---
with open(os.path.join(ART_DIR, "vocab_size.txt")) as f:
    VOCAB_SIZE = int(f.read().strip())

START_TOKEN = VOCAB_SIZE - 2
END_TOKEN   = VOCAB_SIZE - 1
MAX_LEN = 150

with open(os.path.join(ART_DIR, "latex_tokenizer256.pkl"), "rb") as f:
    tokenizer = pkl_load(f)

inv_vocab = {v: k for k, v in tokenizer.word_index.items()}

def decode_tokens(token_ids):
    # Mirror notebook: remove start/end/0 padding, then map to chars
    filtered = [t for t in token_ids if t != START_TOKEN and t != END_TOKEN and t != 0]
    return "".join(inv_vocab.get(t, "") for t in filtered)

def preprocess_pil_to_tensor(pil_img):
    # Mirror notebook preprocessing
    img = pil_img.convert("L").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1], shape (256,256)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
    t = t.repeat(1, 3, 1, 1)  # (1,3,256,256)
    return t.to(DEVICE)

# --- Load model once ---
model = ViTLatexModelLoRA(vocab_size=VOCAB_SIZE).to(DEVICE)
ckpt = torch.load(os.path.join(ART_DIR, "dinov2_attn_lora256.pt"), map_location=DEVICE)

# model.load_state_dict(checkpoint["model"])
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    pil = Image.open(io.BytesIO(file.read())).convert("RGB")

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
    pil.save(os.path.join(SAVE_DIR, filename))

    try:
        x = preprocess_pil_to_tensor(pil)  # (1,3,256,256)
        with torch.no_grad():
            pred_ids = model.generate(x, max_len=MAX_LEN, sos_idx=START_TOKEN, eos_idx=END_TOKEN)
        latex = decode_tokens(pred_ids)
        return jsonify({"latex": latex, "saved": filename})
    except Exception as e:
        return jsonify({"error": str(e), "saved": filename}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)