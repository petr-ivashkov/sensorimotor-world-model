#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
    module load cuda
else
    echo "module command not available; continuing without module load"
fi

cd "$REPO_ROOT"
source "$REPO_ROOT/../.venv/bin/activate"

CONFIG_FILE="${1:-config.yaml}"
python -u "$REPO_ROOT/train.py" --config "$EXPERIMENT_DIR/$CONFIG_FILE" "${@:2}"
