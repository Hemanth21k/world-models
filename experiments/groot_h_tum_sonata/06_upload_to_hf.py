"""
Upload the fine-tuned GR00T-H-N1.7 TUM SonATA checkpoint to HuggingFace Hub.

Uploads the latest (or specified) checkpoint from the fine-tune output directory
along with the model card (MODEL_CARD.md → README.md on the Hub).

Usage:
    python 06_upload_to_hf.py
    CHECKPOINT=checkpoint-20000 HF_REPO=hemanth21k/GR00T-H-N1.7-TUM-SonATA-Franka python 06_upload_to_hf.py
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).parent.parent.parent

HF_REPO = os.environ.get("HF_REPO", "hemanth21k/GR00T-H-N1.7-TUM-SonATA-Franka")
HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUTS_DIR = REPO_ROOT / "outputs" / "groot_h_tum_sonata_finetune"
MODEL_CARD = Path(__file__).parent / "MODEL_CARD.md"

# Pick checkpoint: env var, or latest by step number
CHECKPOINT_NAME = os.environ.get("CHECKPOINT")
if CHECKPOINT_NAME:
    checkpoint_dir = OUTPUTS_DIR / CHECKPOINT_NAME
else:
    checkpoints = sorted(
        [d for d in OUTPUTS_DIR.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {OUTPUTS_DIR}")
    checkpoint_dir = checkpoints[-1]

print(f"Uploading checkpoint: {checkpoint_dir.name}")
print(f"  → HuggingFace repo: {HF_REPO}")

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
create_repo(
    repo_id=HF_REPO,
    repo_type="model",
    exist_ok=True,
    private=False,
    token=HF_TOKEN,
)
print(f"Repo ready: https://huggingface.co/{HF_REPO}")

# Upload model card as README.md
print("Uploading model card...")
api.upload_file(
    path_or_fileobj=str(MODEL_CARD),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Add model card",
)

# Upload checkpoint files
CHECKPOINT_FILES = [
    "config.json",
    "embodiment_id.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
    "trainer_state.json",
]

print(f"Uploading {len(CHECKPOINT_FILES)} checkpoint files...")
for filename in CHECKPOINT_FILES:
    src = checkpoint_dir / filename
    if not src.exists():
        print(f"  WARNING: {filename} not found, skipping")
        continue
    print(f"  {filename} ({src.stat().st_size / 1e9:.2f} GB)" if src.stat().st_size > 1e8 else f"  {filename}")
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=filename,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=f"Upload {checkpoint_dir.name}",
    )

print(f"\nDone. Model available at: https://huggingface.co/{HF_REPO}")
