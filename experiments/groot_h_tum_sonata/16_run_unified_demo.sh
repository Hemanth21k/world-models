#!/bin/bash
# Unified demo: cameras + action curves + error-over-time + robot view + EEF path.
# Two stages (decoupled): 14_rollout_cache.py (GPU, once) → 15_demo_compose.py (CPU).
#
# Iterate on layout fast: set COMPOSE_ONLY=1 to skip the model and just re-render
# from the cached .npz.
#
# Env: GROOT_H_WEIGHTS_DIR, DATASET_DIR, OUTPUTS_DIR, TRAJ_IDS, MODE, GPU, COMPOSE_ONLY
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_finetune}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/h_embodiment_data/Ultrasound/tum/computer_aided_medical_procedures_camp_lab/sonata_all_update/sonata_all}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_demo}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TRAJ_IDS="${TRAJ_IDS:-1916}"
MODE="${MODE:-open_loop}"
GPU="${GPU:-4}"
MAX_STEPS="${MAX_STEPS:-9999}"
COMPOSE_ONLY="${COMPOSE_ONLY:-0}"

[ -f "$REPO_ROOT/datasets/.env" ] && { set -a; source "$REPO_ROOT/datasets/.env"; set +a; }
mkdir -p "$OUTPUTS_DIR"

# Build the list of cache files this run will compose
MODES=$([ "$MODE" = "both" ] && echo "open_loop rollout" || echo "$MODE")
CACHES=""
for T in $TRAJ_IDS; do for M in $MODES; do CACHES="$CACHES /demo_out/cache/rollout_${T}_${M}.npz"; done; done

CACHE_CMD="uv pip install --target=/demo_out/pylibs pybullet >/demo_out/pip_pybullet.log 2>&1"
if [ "$COMPOSE_ONLY" != "1" ]; then
  CACHE_CMD="$CACHE_CMD && uv run --no-sync python /workspace/scripts/14_rollout_cache.py \
        --model-path /checkpoints --dataset-path /data/sonata_all --output-dir /demo_out \
        --traj-ids $TRAJ_IDS --mode $MODE --action-horizon 16 --max-steps $MAX_STEPS --device 0"
fi

docker run --rm --gpus "device=$GPU" --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONPATH=/demo_out/pylibs -e HF_TOKEN="$HF_TOKEN" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$WEIGHTS_DIR":/checkpoints:ro \
    -v "$DATASET_DIR":/data/sonata_all:ro \
    -v "$REPO_ROOT/vla/GR00T-H":/workspace/groot_h \
    -v "$REPO_ROOT/experiments/groot_h_tum_sonata":/workspace/scripts \
    -v "$OUTPUTS_DIR":/demo_out \
    world-models:groot-h-dev \
    bash -c "cd /workspace/groot_h && $CACHE_CMD && \
        uv run --no-sync python /workspace/scripts/15_demo_compose.py \
            --cache $CACHES --output-dir /demo_out --fps 15"

echo "Unified demo videos → $OUTPUTS_DIR"
