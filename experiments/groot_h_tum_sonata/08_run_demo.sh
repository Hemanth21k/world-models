#!/bin/bash
# Generate demo videos: 3-camera + pred vs GT + 3D EEF trajectory.
# Runs 07_demo_video.py inside the groot-h-dev container.
#
# Env overrides:
#   GROOT_H_WEIGHTS_DIR  - fine-tuned checkpoint directory (default: outputs/...)
#   DATASET_DIR          - sonata_all LeRobot dataset
#   OUTPUTS_DIR          - where to write .mp4 files
#   TRAJ_IDS             - space-separated test trajectory IDs (default: 1916 1920 1925)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_finetune}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/h_embodiment_data/Ultrasound/tum/computer_aided_medical_procedures_camp_lab/sonata_all_update/sonata_all}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_demo}"
TRAJ_IDS="${TRAJ_IDS:-1916 1920 1925}"

echo "=== GR00T-H-N1.7 demo video generation ==="
echo "  Weights : $WEIGHTS_DIR"
echo "  Dataset : $DATASET_DIR"
echo "  Outputs : $OUTPUTS_DIR"
echo "  Trajs   : $TRAJ_IDS"
echo "==========================================="

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WEIGHTS_DIR":/outputs:ro \
    -v "$DATASET_DIR":/data/sonata_all:ro \
    -v /fdata1/hemanthp/WorldModelling/world-models/vla/GR00T-H:/workspace/groot_h \
    -v /fdata1/hemanthp/WorldModelling/world-models/experiments/groot_h_tum_sonata:/workspace/scripts \
    -v "$OUTPUTS_DIR":/outputs/demo_videos \
    world-models:groot-h-dev \
    bash -c "cd /workspace/groot_h && uv run python /workspace/scripts/07_demo_video.py \
        --model-path /outputs \
        --dataset-path /data/sonata_all \
        --output-dir /outputs/demo_videos \
        --traj-ids $TRAJ_IDS \
        --mode both \
        --action-horizon 16 \
        --max-steps 300 \
        --fps 10"

echo ""
echo "Videos saved to: $OUTPUTS_DIR"
