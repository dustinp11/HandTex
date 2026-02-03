# notebooks/eda.ipynb (or whatever notebook)
from datasets import load_dataset
import os
from pathlib import Path

# Get project root directory (parent of notebooks/)
project_root = Path.cwd().parent
data_dir = project_root / "data"

data_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("deepcopy/MathWriting-human")
dataset.save_to_disk(str(data_dir / "mathwriting"))