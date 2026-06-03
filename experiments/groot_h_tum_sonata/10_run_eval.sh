#!/bin/bash
# Run evaluation sweep over test split.
# Requires training to have finished (uses final checkpoint by default).
#
# Env overrides:
#   GROOT_H_WEIGHTS_DIR  - fine-tuned checkpoint dir  (default: outputs/finetune)
#   DATASET_DIR          - sonata_all LeRobot dataset  (default: datasets/...)
#   EVAL_OUT             - where to write eval_results.json
#   HORIZONS             - space-separated horizon values  (default: "1 4 8 16 50")
#   MODES                - "open_loop rollout" or just one  (default: both)
#   MAX_EPISODES         - cap test episodes for a quick run  (default: all 482)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_finetune}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/h_embodiment_data/Ultrasound/tum/computer_aided_medical_procedures_camp_lab/sonata_all_update/sonata_all}"
EVAL_OUT="${EVAL_OUT:-$REPO_ROOT/outputs/groot_h_tum_sonata_eval}"
HORIZONS="${HORIZONS:-1 4 8 16 50}"
MODES="${MODES:-open_loop rollout}"
MAX_EPISODES="${MAX_EPISODES:-}"

mkdir -p "$EVAL_OUT"

echo "=== GR00T-H-N1.7 TUM SonATA Franka evaluation ==="
echo "  Weights  : $WEIGHTS_DIR"
echo "  Dataset  : $DATASET_DIR"
echo "  Output   : $EVAL_OUT"
echo "  Horizons : $HORIZONS"
echo "  Modes    : $MODES"
echo "  Max eps  : ${MAX_EPISODES:-all}"
echo "=================================================="

MAX_EPISODES_ARG=""
[ -n "$MAX_EPISODES" ] && MAX_EPISODES_ARG="--max-episodes $MAX_EPISODES"

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HF_TOKEN="${HF_TOKEN}" \
    -v "$WEIGHTS_DIR":/checkpoints:ro \
    -v "$DATASET_DIR":/data/sonata_all:ro \
    -v /fdata1/hemanthp/WorldModelling/world-models/vla/GR00T-H:/workspace/groot_h \
    -v /fdata1/hemanthp/WorldModelling/world-models/experiments/groot_h_tum_sonata:/workspace/scripts \
    -v "$EVAL_OUT":/eval_out \
    world-models:groot-h-dev \
    bash -c "cd /workspace/groot_h && uv run python /workspace/scripts/09_eval.py \
        --model-path /checkpoints \
        --dataset-path /data/sonata_all \
        --output-dir /eval_out \
        --horizons $HORIZONS \
        --modes $MODES \
        $MAX_EPISODES_ARG"

echo ""
echo "Results: $EVAL_OUT/eval_results.json"
