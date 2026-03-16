# HandTeX — Handwriting to LaTeX
# Course project for UCI CS 175 (Winter 2026)
# Encoder-decoder models that convert handwritten math images to LaTeX strings.
# Dataset: deepcopy/MathWriting-human on HuggingFace (~230k samples).

# External libraries used
Libraries used:
    - PyTorch (https://pytorch.org/)
    - torchvision (https://pytorch.org/vision/)
    - HuggingFace Transformers (https://github.com/huggingface/transformers)
    - HuggingFace Datasets (https://github.com/huggingface/datasets)
    - HuggingFace PEFT / LoRA (https://github.com/huggingface/peft)
    - Pillow (https://python-pillow.org/)
    - Flask (https://flask.palletsprojects.com/)
    - TensorFlow/Keras (https://www.tensorflow.org/) — used only for tokenizer utilities
    - NumPy (https://numpy.org/)

# Publicly available code(s) used in this project
Publicly available codes used:
    - facebook/dinov2-base pretrained weights (https://huggingface.co/facebook/dinov2-base). Not modified, used as the frozen encoder backbone with LoRA adapters applied on top.

# Code written entirely by our team
Scripts/functions written by our team:
    - src/models/vit_lora_lstm_attn.py  DINOv2 encoder with LoRA + LSTM decoder with Bahdanau attention, includes greedy and beam search decoding (255 lines)
    - src/models/vit_transformer_v2.py  DINOv2 encoder with LoRA + Transformer decoder (67 lines)
    - backend/app.py  Flask API for model inference from the web/mobile frontend (98 lines)
    - get_data.py  Downloads the MathWriting dataset from HuggingFace and saves locally (12 lines)
    - project.ipynb  Main evaluation notebook: loads dataset, loads models, runs evaluation on the test set (~120 lines of code across 14 cells)
    - src/evals.ipynb  Evaluation notebook comparing LSTM, Transformer, and GPT-5.2 baselines on the validation set (~200 lines of code across 11 cells)
    - src/evals_newdata.ipynb  Evaluation on CROHME external dataset (~130 lines of code across 7 cells)
    - src/partition.ipynb  Annotation tool and evaluation by handwriting difficulty (easy/medium/hard partitions) (~180 lines of code across 8 cells)
