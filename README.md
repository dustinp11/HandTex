# HandTeX

**Handwriting-to-LaTeX converter.** Snap a photo of a handwritten math equation and get clean LaTeX back in real time.

## Demo

[![HandTeX Demo](https://img.youtube.com/vi/VNxEOfVlMCM/maxresdefault.jpg)](https://www.youtube.com/watch?v=VNxEOfVlMCM&list=PLPac3mAhH0PhFlXR-guO7qnDi3EEuRpSs&index=2)

*Click the thumbnail to watch the demo on YouTube.*

## What it does

HandTeX converts images of handwritten mathematical equations into LaTeX strings using a vision encoder-decoder architecture. Trained on 230K samples from the MathWriting dataset, it achieves **50% exact match accuracy** on the held-out test set with real-time inference.

## Architecture

- **Encoder**: Frozen DINOv2-base with LoRA adapters for efficient fine-tuning
- **Decoder**: Custom LSTM with Bahdanau attention (primary) + Transformer decoder variant
- **Training**: Teacher-forced with CrossEntropyLoss, character-level tokenization
- **Inference**: Greedy autoregressive decoding (beam search also implemented)

## Tech Stack

PyTorch · HuggingFace Transformers · PEFT/LoRA · Flask · React Native

## Project Structure

```
handtex/
├── src/
│   ├── models/
│   │   ├── vit_lora_lstm_attn.py        # DINOv2 + LoRA + LSTM w/ Bahdanau attention (primary model)
│   │   ├── vit_transformer_v2.py        # DINOv2 + LoRA + Transformer decoder
│   │   ├── vit-transformer.py           # ViT + Transformer decoder
│   │   └── vit-transformer-lora-r16.py  # ViT + Transformer decoder w/ LoRA r=16
│   ├── vit.ipynb                        # ViT encoder experiments
│   ├── vitencoder-dinov2-2.ipynb        # DINOv2 encoder training
│   ├── cnn.ipynb                        # CNN baseline
│   ├── evals.ipynb                      # Evaluation vs. GPT-5.2 baseline
│   ├── evals_newdata.ipynb              # External eval on CROHME dataset
│   └── partition.ipynb                  # Difficulty-partitioned eval (easy/medium/hard)
├── backend/
│   ├── app.py                           # Flask API — POST /predict, returns {"latex": ...}
│   ├── models/
│   │   └── vit_lora_lstm_attn.py        # Model copy for inference
│   └── artifacts/
│       └── vocab_size.txt
├── mobile/
│   ├── App.js                           # React Native app
│   ├── app.json
│   └── assets/
├── output/
│   └── gpt5_baseline_results.json
├── get_data.py                          # Downloads MathWriting dataset from HuggingFace
├── requirements.txt
└── README.md
```

## Dataset

[MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human) — ~230K handwritten math equations. External validation on CROHME dataset.

## Course

Built for UCI CS 175 (Winter 2026) as a 3-person team project.

## Attribution

- **Pretrained weights**: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base), used as frozen backbone with LoRA adapters
- **Libraries**: PyTorch, torchvision, HuggingFace Transformers/Datasets/PEFT, Pillow, Flask, NumPy (TensorFlow/Keras used only for tokenizer utilities)
- All model code, training pipeline, evaluation notebooks, and Flask API written by the team
