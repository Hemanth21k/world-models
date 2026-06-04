#!/bin/bash
# Robot rollout video: Franka executing GR00T-H-N1.7's predicted EEF poses
# (GT ghost overlaid with predicted arm) via PyBullet. Runs 12_robot_rollout.py
# inside the groot-h-dev container.
#
# PyBullet is installed into an ISOLATED path (OUTPUTS_DIR/pylibs) so it never
# mutates the shared container venv — safe to run alongside other jobs.
#
# Env overrides: GROOT_H_WEIGHTS_DIR, DATASET_DIR, OUTPUTS_DIR, TRAJ_IDS, GPU
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_finetune}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/h_embodiment_data/Ultrasound/tum/computer_aided_medical_procedures_camp_lab/sonata_all_update/sonata_all}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_demo}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TRAJ_IDS="${TRAJ_IDS:-1916}"
GPU="${GPU:-4}"
MODE="${MODE:-open_loop}"
MAX_STEPS="${MAX_STEPS:-9999}"

# HF_TOKEN needed to load the gated Cosmos-Reason2-2B backbone config
[ -f "$REPO_ROOT/datasets/.env" ] && { set -a; source "$REPO_ROOT/datasets/.env"; set +a; }

mkdir -p "$OUTPUTS_DIR"
echo "=== GR00T-H-N1.7 robot rollout (PyBullet) ==="
echo "  Weights:$WEIGHTS_DIR"; echo "  Trajs:$TRAJ_IDS  GPU:$GPU  mode:$MODE"

docker run --rm --gpus "device=$GPU" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONPATH=/demo_out/pylibs \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$WEIGHTS_DIR":/checkpoints:ro \
    -v "$DATASET_DIR":/data/sonata_all:ro \
    -v "$REPO_ROOT/vla/GR00T-H":/workspace/groot_h \
    -v "$REPO_ROOT/experiments/groot_h_tum_sonata":/workspace/scripts \
    -v "$OUTPUTS_DIR":/demo_out \
    world-models:groot-h-dev \
    bash -c "cd /workspace/groot_h && \
        uv pip install --target=/demo_out/pylibs pybullet >/demo_out/pip_pybullet.log 2>&1 && \
        uv run --no-sync python /workspace/scripts/12_robot_rollout.py \
            --model-path /checkpoints --dataset-path /data/sonata_all \
            --output-dir /demo_out --traj-ids $TRAJ_IDS \
            --mode $MODE --action-horizon 16 --max-steps $MAX_STEPS --fps 15 --device 0"

echo "Robot rollout videos → $OUTPUTS_DIR"
