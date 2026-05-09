#!/usr/bin/env bash
# Unified helper for building and running world-models containers.
# Run from repo root: bash docker/docker_run.sh <command> <model> [args...]
#
# Models:  groot-h | groot-h-dev | vjepa2 | vjepa2-dev | all
# Modes:   groot-h      → deployment (code baked in)
#          groot-h-dev  → development (deps only, code mounted from repo)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker-compose.yml"

CMD=${1:-help}
MODEL=${2:-all}

case "$CMD" in

  build)
    if [ "$MODEL" = "all" ]; then
      echo "Building all images (deployment + dev)..."
      $COMPOSE build
    else
      echo "Building world-models:$MODEL ..."
      $COMPOSE build "$MODEL"
    fi
    ;;

  shell)
    [ "$MODEL" = "all" ] && { echo "Specify a model: groot-h | groot-h-dev | vjepa2 | vjepa2-dev"; exit 1; }
    echo "Starting shell in $MODEL container..."
    if [[ "$MODEL" == *-dev ]]; then
      BASE="${MODEL%-dev}"
      echo "  Dev mode: $BASE source is mounted from the repo."
      echo "  Run inside: pip install -e /workspace/${BASE//-/_} --no-deps"
    fi
    $COMPOSE run --rm "$MODEL" /bin/bash
    ;;

  gpu-check)
    TARGET=${MODEL:-groot-h}
    echo "Checking GPU visibility in $TARGET container..."
    $COMPOSE run --rm "$TARGET" python -c \
      "import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}'); [print(f'  [{i}]', torch.cuda.get_device_name(i)) for i in range(n)]"
    ;;

  run)
    [ "$MODEL" = "all" ] && { echo "Specify a model."; exit 1; }
    shift 2
    $COMPOSE run --rm "$MODEL" "$@"
    ;;

  *)
    cat <<'EOF'
Usage: bash docker/docker_run.sh <command> [model] [args...]

Models:
  groot-h        Deployment — GR00T-H source baked in
  groot-h-dev    Development — deps only, source mounted from repo
  vjepa2         Deployment — V-JEPA 2 source baked in
  vjepa2-dev     Development — deps only, source mounted from repo
  all            All of the above (build only)

Commands:
  build  [model|all]    Build image(s)
  shell   model         Open interactive bash shell
  gpu-check [model]     Verify all GPUs are visible
  run   model <cmd...>  Run an arbitrary command in the container

Environment variables:
  WEIGHTS_DIR   Path to downloaded model weights   (default: ~/models/GR00T-H)
  DATASET_DIR   Path to dataset                    (required for inference)

Examples:
  bash docker/docker_run.sh build groot-h
  bash docker/docker_run.sh build all

  # Deployment shell (code baked in)
  bash docker/docker_run.sh shell groot-h

  # Development shell (live code mount, then install inside)
  bash docker/docker_run.sh shell groot-h-dev
  # → inside container: pip install -e /workspace/groot_h --no-deps

  DATASET_DIR=/data/sonata_all bash docker/docker_run.sh shell vjepa2
  bash docker/docker_run.sh gpu-check vjepa2-dev
EOF
    ;;
esac
