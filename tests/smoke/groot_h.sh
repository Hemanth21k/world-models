#!/usr/bin/env bash
# Smoke test for world-models:groot-h
# Runs inside the container — checks GPU visibility and package import.
# Usage: bash tests/smoke/groot_h.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker/docker-compose.yml"

echo "=== Smoke test: world-models:groot-h ==="

echo "[1/2] GPU check..."
$COMPOSE run --rm groot-h python -c "
import torch
n = torch.cuda.device_count()
assert n > 0, 'No GPUs found!'
print(f'  PASS — {n} GPU(s) visible: {[torch.cuda.get_device_name(i) for i in range(n)]}')
"

echo "[2/2] Package import..."
$COMPOSE run --rm groot-h python -c "
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag
print('  PASS — gr00t package imports successfully')
print(f'  TUM_SONATA_FRANKA tag: {EmbodimentTag.TUM_SONATA_FRANKA}')
"

echo ""
echo "=== groot-h smoke test PASSED ==="
echo "Update tests/STATUS.md checkboxes for: Docker build, GPU, Package import"
