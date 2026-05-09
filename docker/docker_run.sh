#!/usr/bin/env bash
# Unified helper for building and running world-models containers.
# Run from repo root: bash docker/docker_run.sh <command> [model] [args...]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker-compose.yml"

CMD=${1:-help}
MODEL=${2:-all}

case "$CMD" in

  build)
    if [ "$MODEL" = "all" ]; then
      echo "Building all model images..."
      $COMPOSE build
    else
      echo "Building world-models:$MODEL ..."
      $COMPOSE build "$MODEL"
    fi
    ;;

  shell)
    [ "$MODEL" = "all" ] && { echo "Specify a model: groot-h | vjepa2"; exit 1; }
    echo "Starting interactive shell in $MODEL container..."
    $COMPOSE run --rm "$MODEL" /bin/bash
    ;;

  gpu-check)
    TARGET=${MODEL:-groot-h}
    echo "Checking GPU visibility in $TARGET container..."
    $COMPOSE run --rm "$TARGET" python -c \
      "import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}'); [print(f'  [{i}]', torch.cuda.get_device_name(i)) for i in range(n)]"
    ;;

  run)
    # Generic: docker/docker_run.sh run <model> <command...>
    [ "$MODEL" = "all" ] && { echo "Specify a model: groot-h | vjepa2"; exit 1; }
    shift 2
    $COMPOSE run --rm "$MODEL" "$@"
    ;;

  *)
    cat <<EOF
Usage: bash docker/docker_run.sh <command> [model] [extra args]

Commands:
  build  [groot-h|vjepa2|all]   Build image(s)                (default: all)
  shell   groot-h|vjepa2         Open interactive bash shell
  gpu-check [model]              Verify all GPUs are visible
  run     model  <cmd...>        Run an arbitrary command in the container

Environment variables (override via export or prefix):
  WEIGHTS_DIR    Path to downloaded model weights  (default: ~/models/GR00T-H)
  DATASET_DIR    Path to dataset                   (required for inference)

Examples:
  bash docker/docker_run.sh build groot-h
  DATASET_DIR=/data/sonata_all bash docker/docker_run.sh shell groot-h
  bash docker/docker_run.sh gpu-check vjepa2
EOF
    ;;
esac
