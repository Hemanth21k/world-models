#!/usr/bin/env bash
# Smoke test for world-models:vjepa2
# Runs inside the container — checks GPU visibility and package import.
# Usage: bash tests/smoke/vjepa2.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE="docker compose -f $SCRIPT_DIR/docker/docker-compose.yml"

echo "=== Smoke test: world-models:vjepa2 ==="

echo "[1/2] GPU check..."
$COMPOSE run --rm vjepa2 python -c "
import torch
n = torch.cuda.device_count()
assert n > 0, 'No GPUs found!'
print(f'  PASS — {n} GPU(s) visible: {[torch.cuda.get_device_name(i) for i in range(n)]}')
"

echo "[2/2] Package import..."
$COMPOSE run --rm vjepa2 python -c "
import app
import src.models
print('  PASS — vjepa2 package imports successfully')
"

echo ""
echo "=== vjepa2 smoke test PASSED ==="
echo "Update tests/STATUS.md checkboxes for: Docker build, GPU, Package import"
