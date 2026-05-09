#!/bin/bash
# Build the gr00t-dev Docker image using the existing build script in the submodule.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$REPO_ROOT/vla/GR00T-H/docker/build.sh" "$@"
