# HandTeX

> **WIP**

Handwriting-to-LaTeX: draw a math equation on a canvas and get the LaTeX back. Built with a pretrained Vision Transformer (ViT) encoder + custom LSTM decoder trained on the [MathWriting](https://huggingface.co/datasets/deepcopy/MathWriting-human). Training is done through Kaggle notebooks. 


Course project for UCI CS 175 (Winter 2026).

## Structure

```
handtex/
  backend/     Flask API for model inference
  models/      ViT model definition
  notebooks/   Model training and saving 
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

## Kaggle Training

1) Add Notebook (vit.ipynb for example) into a kaggle notebook with accelerator set to a GPU. 
2) Zip the mathwriting dataset and add as input. 
3) Change the paths in the notebook to match. 
4) Save version and run all to run in background. 

