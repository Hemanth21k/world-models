#!/bin/bash
# Generate temporal_stats.json for the TUM SonATA Franka dataset.
# Must be run before fine-tuning (05_finetune.sh).
#
# This step reads all parquet files (no video decoding), converts actions to
# REL_XYZ_ROT6D, and computes per-timestep mean/std over the 50-step horizon.
# Output: <DATASET_DIR>/meta/temporal_stats.json
#
# Env overrides:
#   GROOT_H_WEIGHTS_DIR  - path to GR00T-H-N1.7 weights (default below)
#   DATASET_DIR          - path to sonata_all LeRobot dataset (default below)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$REPO_ROOT/weights/GR00T-H-N1.7}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/h_embodiment_data/Ultrasound/tum/computer_aided_medical_procedures_camp_lab/sonata_all_update/sonata_all}"

echo "=== GR00T-H TUM SonATA Franka — stats generation ==="
echo "  Weights : $WEIGHTS_DIR"
echo "  Dataset : $DATASET_DIR"
echo "====================================================="

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WEIGHTS_DIR":/weights/GR00T-H-N1.7:ro \
    -v "$DATASET_DIR":/data/sonata_all \
    world-models:groot-h-dev \
    bash -c "cd /workspace/groot_h && uv run python gr00t/experiment/launch_finetune.py \
        --base-model-path /weights/GR00T-H-N1.7 \
        --dataset-path /data/sonata_all \
        --embodiment-tag TUM_SONATA_FRANKA \
        --calculate-norm-stats"

echo ""
echo "temporal_stats.json written to: $DATASET_DIR/meta/"
