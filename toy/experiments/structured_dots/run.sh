#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/../.venv/bin/activate"

CONFIG_FILE="${1:-config_2_independent.yaml}"
python -u "$REPO_ROOT/train.py" --config "$EXPERIMENT_DIR/$CONFIG_FILE" "${@:2}"
