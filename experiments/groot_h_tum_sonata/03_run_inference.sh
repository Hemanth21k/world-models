#!/bin/bash
# Run GR00T-H inference on TUM SonATA Franka data inside the gr00t-dev container.
# Plots (pred vs ground truth per action dim) are saved to OUTPUTS_DIR.
#
# Env overrides:
#   GROOT_H_WEIGHTS_DIR  - local path to downloaded nvidia/GR00T-H weights
#   DATASET_DIR          - local path to sonata_all LeRobot dataset
#   OUTPUTS_DIR          - where to write traj_*.jpeg plots
#   TRAJ_IDS             - space-separated trajectory IDs (default: "0 1 2")
#   ACTION_HORIZON       - steps per inference call (default: 50, matches TUM config)
#   STEPS                - max steps per trajectory (default: 300)
#   VIDEO_BACKEND        - torchcodec | torchvision_av | decord (default: torchcodec)
#   DENOISING_STEPS      - diffusion denoising steps (default: 4)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$HOME/models/GR00T-H}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR to your local sonata_all LeRobot dataset path}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata}"
TRAJ_IDS="${TRAJ_IDS:-0 1 2}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
STEPS="${STEPS:-300}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"

mkdir -p "$OUTPUTS_DIR"

echo "=== GR00T-H TUM SonATA Franka inference ==="
echo "  Weights   : $WEIGHTS_DIR"
echo "  Dataset   : $DATASET_DIR"
echo "  Outputs   : $OUTPUTS_DIR"
echo "  Traj IDs  : $TRAJ_IDS"
echo "  Horizon   : $ACTION_HORIZON steps  (model predicts 50, re-infers every $ACTION_HORIZON)"
echo "  Steps     : $STEPS per trajectory"
echo "============================================"

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WEIGHTS_DIR":/weights/GR00T-H:ro \
    -v "$DATASET_DIR":/data/sonata_all:ro \
    -v "$OUTPUTS_DIR":/tmp/stand_alone_inference \
    world-models:groot-h \
    bash -c "python scripts/deployment/standalone_inference_script.py \
        --model-path /weights/GR00T-H \
        --dataset-path /data/sonata_all \
        --embodiment-tag TUM_SONATA_FRANKA \
        --traj-ids $TRAJ_IDS \
        --action-horizon $ACTION_HORIZON \
        --steps $STEPS \
        --denoising-steps $DENOISING_STEPS \
        --inference-mode pytorch \
        --video-backend $VIDEO_BACKEND"

echo ""
echo "Plots saved to: $OUTPUTS_DIR"
