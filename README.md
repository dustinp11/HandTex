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

## My Role

Team lead for 3-person team (UCI CS 175, Winter 2026). Led model architecture design, wrote the core LSTM + attention decoder, built the Flask inference API, and ran evaluation against GPT-5.2 and Claude 4.5 Sonnet baselines.

## Project Structure

```
src/models/
  vit_lora_lstm_attn.py     # DINOv2 + LoRA encoder, LSTM decoder w/ Bahdanau attention
  vit_transformer_v2.py     # Transformer decoder variant
backend/
  app.py                    # Flask inference API
src/
  evals.ipynb               # Evaluation vs. LSTM/Transformer/GPT-5.2 baselines
  evals_newdata.ipynb       # External eval on CROHME dataset
  partition.ipynb           # Difficulty-partitioned eval (easy/medium/hard)
project.ipynb               # Main evaluation notebook
get_data.py                 # MathWriting dataset download script
```

## Dataset

[MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human) — ~230K handwritten math equations. External validation on CROHME dataset.

## Attribution

- **Pretrained weights**: [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base), used as frozen backbone with LoRA adapters
- **Libraries**: PyTorch, torchvision, HuggingFace Transformers/Datasets/PEFT, Pillow, Flask, NumPy (TensorFlow/Keras used only for tokenizer utilities)
- All model code, training pipeline, evaluation notebooks, and Flask API written by our team
