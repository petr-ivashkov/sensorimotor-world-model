#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/../.venv/bin/activate"

# One config as $1 (e.g. config_single_dot_kmax4.yaml), or all of them if omitted.
if [[ $# -ge 1 ]]; then
    python -u "$REPO_ROOT/train_multistep.py" --config "$EXPERIMENT_DIR/$1" "${@:2}"
else
    for cfg in "$EXPERIMENT_DIR"/config_*_kmax*.yaml; do
        python -u "$REPO_ROOT/train_multistep.py" --config "$cfg"
    done
fi
