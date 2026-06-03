"""
Download nvidia/GR00T-H-N1.7 weights from HuggingFace.

Usage:
    python 02_download_weights.py
    GROOT_H_WEIGHTS_DIR=/my/path HF_TOKEN=hf_xxx python 02_download_weights.py
"""
import os
from huggingface_hub import snapshot_download

WEIGHTS_DIR = os.environ.get(
    "GROOT_H_WEIGHTS_DIR",
    os.path.join(os.path.expanduser("~"), "models", "GR00T-H-N1.7"),
)
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Downloading nvidia/GR00T-H-N1.7 → {WEIGHTS_DIR}")
snapshot_download(
    repo_id="nvidia/GR00T-H-N1.7",
    local_dir=WEIGHTS_DIR,
    token=HF_TOKEN,
)
print(f"Done. Weights at: {WEIGHTS_DIR}")
