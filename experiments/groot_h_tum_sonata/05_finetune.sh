#!/bin/bash
# Fine-tune GR00T-H-N1.7 on TUM SonATA Franka (train split only).
# Runs inside world-models:groot-h-dev via Docker, detached under nohup.
#
# Prerequisites:
#   1. Docker image built:  bash docker/docker_run.sh build groot-h-dev
#   2. Stats generated:     bash experiments/groot_h_tum_sonata/04_prep_stats.sh
#
# Env overrides:
#   GROOT_H_WEIGHTS_DIR  - path to GR00T-H-N1.7 weights
#   DATASET_DIR          - path to sonata_all LeRobot dataset
#   OUTPUTS_DIR          - where to write checkpoints (default: outputs/groot_h_tum_sonata_finetune)
#   MAX_STEPS            - training steps (default: 20000)
#   NUM_GPUS             - number of GPUs to use (default: 6)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WEIGHTS_DIR="${GROOT_H_WEIGHTS_DIR:-$REPO_ROOT/weights/GR00T-H-N1.7}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/h_embodiment_data/Ultrasound/tum/computer_aided_medical_procedures_camp_lab/sonata_all_update/sonata_all}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_ROOT/outputs/groot_h_tum_sonata_finetune}"
MAX_STEPS="${MAX_STEPS:-20000}"
NUM_GPUS="${NUM_GPUS:-6}"
GLOBAL_BATCH_SIZE=$(( NUM_GPUS * 4 ))  # 4 per GPU

mkdir -p "$OUTPUTS_DIR"
LOG_FILE="$OUTPUTS_DIR/finetune.log"

echo "=== GR00T-H-N1.7 TUM SonATA Franka fine-tune ==="
echo "  Weights     : $WEIGHTS_DIR"
echo "  Dataset     : $DATASET_DIR"
echo "  Outputs     : $OUTPUTS_DIR"
echo "  GPUs        : $NUM_GPUS"
echo "  Batch size  : $GLOBAL_BATCH_SIZE (${NUM_GPUS} GPUs × 4)"
echo "  Max steps   : $MAX_STEPS"
echo "  Log file    : $LOG_FILE"
echo "=================================================="
echo ""
echo "Launching in background. Monitor with:"
echo "  tail -f $LOG_FILE"
echo ""

nohup docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$WEIGHTS_DIR":/weights/GR00T-H-N1.7:ro \
    -v "$DATASET_DIR":/data/sonata_all:ro \
    -v "$OUTPUTS_DIR":/outputs \
    world-models:groot-h-dev \
    bash -c "cd /workspace/groot_h && uv run torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        gr00t/experiment/launch_finetune.py \
        --base-model-path /weights/GR00T-H-N1.7 \
        --dataset-path /data/sonata_all \
        --embodiment-tag TUM_SONATA_FRANKA \
        --num-gpus $NUM_GPUS \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --max-steps $MAX_STEPS \
        --include-splits train \
        --state-dropout-prob 0.0 \
        --output-dir /outputs \
        --save-steps 1000 \
        --save-total-limit 5 \
        --use-wandb False" \
    >> "$LOG_FILE" 2>&1 &

echo "Training PID: $!"
echo "$!" > "$OUTPUTS_DIR/finetune.pid"
echo "To stop: kill \$(cat $OUTPUTS_DIR/finetune.pid)"
