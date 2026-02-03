# HandTeX

> **WIP** -- everything here is subject to change.

Handwriting-to-LaTeX: draw a math equation on a canvas and get the LaTeX back. Built with a Vision Transformer (ViT) trained on the [MathWriting](https://huggingface.co/datasets/deepcopy/MathWriting-human) dataset.

Course project for UCI CS 175 (Winter 2026).

## Structure

```
handtex/
  backend/     Flask API for model inference
  models/      ViT model definition
  train/       Training scripts
  notebooks/   EDA and experimentation
  frontend/    HTML5 Canvas UI
  data/        Dataset (git-ignored, see below)
```

## Getting the Data

The dataset is not included in the repo. It's hosted on HuggingFace as [deepcopy/MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human) (~1.4 GB download, ~230k training examples of handwritten math images paired with LaTeX strings).

To download and save it locally, just run get_data.py file. 


This will create `data/mathwriting/` with train, test, and val splits in Arrow format. The `data/` directory is git-ignored.

## Setup

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.12+
- PyTorch
- HuggingFace `datasets`
- Pillow
- Flask

## Status

Early development. Model, training loop, backend API, and frontend are all stubbed out.
