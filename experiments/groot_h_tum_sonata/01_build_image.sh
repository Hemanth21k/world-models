#!/bin/bash
# Build the world-models:groot-h Docker image using the unified Dockerfile.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$REPO_ROOT/docker/docker_run.sh" build groot-h "$@"
