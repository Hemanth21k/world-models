#!/usr/bin/env bash
# Unified helper for building and running world-models containers.
# Works with any model service defined in docker/docker-compose.yml.
#
# All images share the "world-models" repository name and are tagged by model:
#   world-models:groot-h       deployment (source baked in)
#   world-models:groot-h-dev   development (source mounted live)
#   world-models:vjepa2        deployment
#   world-models:vjepa2-dev    development
#   ... (any service added to docker-compose.yml)
#
# Run from repo root:
#   bash docker/docker_run.sh <command> [model] [args...]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker-compose.yml"

CMD=${1:-help}
MODEL=${2:-}

case "$CMD" in

  list)
    echo "Available world-models services:"
    echo ""
    echo "  DEPLOYMENT (source baked in):"
    $COMPOSE config --services | grep -v '\-dev' | sort | sed 's/^/    world-models:/'
    echo ""
    echo "  DEVELOPMENT (source mounted live):"
    $COMPOSE config --services | grep '\-dev'  | sort | sed 's/^/    world-models:/'
    ;;

  build)
    if [ -z "$MODEL" ] || [ "$MODEL" = "all" ]; then
      echo "Building all world-models images..."
      $COMPOSE build
    else
      echo "Building world-models:$MODEL ..."
      $COMPOSE build "$MODEL"
    fi
    ;;

  shell)
    if [ -z "$MODEL" ]; then
      echo "Specify a model tag. Available services:"
      $COMPOSE config --services | sort | sed 's/^/  /'
      exit 1
    fi
    echo "Starting shell in world-models:$MODEL ..."
    if [[ "$MODEL" == *-dev ]]; then
      echo "  Dev mode: source is mounted live from the repo."
      echo "  Run inside the container once: pip install -e . --no-deps"
    fi
    $COMPOSE run --rm "$MODEL" /bin/bash
    ;;

  gpu-check)
    if [ -z "$MODEL" ]; then
      MODEL=$($COMPOSE config --services | grep -v '\-dev' | head -1)
    fi
    echo "Checking GPU visibility in world-models:$MODEL ..."
    $COMPOSE run --rm "$MODEL" python -c \
      "import torch; n=torch.cuda.device_count(); print(f'GPUs: {n}'); [print(f'  [{i}]', torch.cuda.get_device_name(i)) for i in range(n)]"
    ;;

  run)
    if [ -z "$MODEL" ]; then
      echo "Specify a model tag. Run: bash docker/docker_run.sh list"; exit 1
    fi
    shift 2
    $COMPOSE run --rm "$MODEL" "$@"
    ;;

  *)
    cat <<'EOF'
Usage: bash docker/docker_run.sh <command> [model-tag] [args...]

All images are tagged as  world-models:<model-tag>  (e.g. world-models:groot-h).
The "world-models" repository name is always the parent — tags identify models.

Commands:
  list                      Show all available model tags (deployment + dev)
  build  [model-tag|all]    Build one image or all images       (default: all)
  shell   model-tag         Open an interactive bash shell
  gpu-check [model-tag]     Verify all GPUs are visible
  run   model-tag <cmd...>  Run any command inside a container

Model tag convention (defined in docker/docker-compose.yml):
  <model>      Deployment — source baked in, no mounts needed
  <model>-dev  Development — deps only, source mounted live from repo

Environment variables:
  WEIGHTS_DIR   Path to model weights  (default: ~/models/<model>)
  DATASET_DIR   Path to dataset        (required for inference)

Examples:
  bash docker/docker_run.sh list
  bash docker/docker_run.sh build groot-h
  bash docker/docker_run.sh build                        # builds everything
  bash docker/docker_run.sh shell groot-h                # deployment shell
  bash docker/docker_run.sh shell groot-h-dev            # dev shell
  bash docker/docker_run.sh gpu-check                    # uses first service
  DATASET_DIR=/data bash docker/docker_run.sh run vjepa2 python app/main.py
EOF
    ;;
esac
