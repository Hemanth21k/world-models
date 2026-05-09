#!/usr/bin/env bash
# Unified helper for building and running world-models containers.
# Works with any model service defined in docker/docker-compose.yml.
#
# Run from repo root:
#   bash docker/docker_run.sh <command> [model] [args...]
#
# Convention (defined in docker-compose.yml):
#   <model>      → deployment image  (source baked in, no mounts needed)
#   <model>-dev  → development image (deps only, source mounted live)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker-compose.yml"

CMD=${1:-help}
MODEL=${2:-all}

case "$CMD" in

  list)
    echo "Available model services:"
    $COMPOSE config --services | sort
    ;;

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
    [ "$MODEL" = "all" ] && { echo "Specify a model. Run: bash docker/docker_run.sh list"; exit 1; }
    echo "Starting shell in $MODEL container..."
    if [[ "$MODEL" == *-dev ]]; then
      echo "  Dev mode: source is mounted live from the repo."
      echo "  Run inside the container once: pip install -e . --no-deps"
    fi
    $COMPOSE run --rm "$MODEL" /bin/bash
    ;;

  gpu-check)
    if [ "$MODEL" = "all" ]; then
      MODEL=$($COMPOSE config --services | grep -v '\-dev' | head -1)
    fi
    echo "Checking GPU visibility in $MODEL container..."
    $COMPOSE run --rm "$MODEL" python -c \
      "import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}'); [print(f'  [{i}]', torch.cuda.get_device_name(i)) for i in range(n)]"
    ;;

  run)
    [ "$MODEL" = "all" ] && { echo "Specify a model. Run: bash docker/docker_run.sh list"; exit 1; }
    shift 2
    $COMPOSE run --rm "$MODEL" "$@"
    ;;

  *)
    cat <<'EOF'
Usage: bash docker/docker_run.sh <command> [model] [args...]

Commands:
  list                    List all available model services
  build  [model|all]      Build image(s)
  shell   model           Open interactive bash shell
  gpu-check [model]       Verify all GPUs are visible
  run   model <cmd...>    Run an arbitrary command in the container

Model convention (services defined in docker/docker-compose.yml):
  <model>      Deployment — source baked in, fully self-contained
  <model>-dev  Development — deps only, source mounted live from repo

Environment variables:
  WEIGHTS_DIR   Path to model weights  (default: ~/models/<model>)
  DATASET_DIR   Path to dataset        (required for inference)

Examples:
  bash docker/docker_run.sh list
  bash docker/docker_run.sh build groot-h
  bash docker/docker_run.sh build all
  bash docker/docker_run.sh shell groot-h
  bash docker/docker_run.sh shell groot-h-dev
  bash docker/docker_run.sh gpu-check
  DATASET_DIR=/data bash docker/docker_run.sh run vjepa2 python app/main.py
EOF
    ;;
esac
