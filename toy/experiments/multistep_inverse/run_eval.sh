#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/../.venv/bin/activate"

# One eval config as $1 (e.g. config_eval_single_dot.yaml), or all if omitted.
if [[ $# -ge 1 ]]; then
    python -u "$REPO_ROOT/eval_goal_reaching.py" --config "$EXPERIMENT_DIR/$1" "${@:2}"
else
    for cfg in "$EXPERIMENT_DIR"/config_eval_*.yaml; do
        python -u "$REPO_ROOT/eval_goal_reaching.py" --config "$cfg"
    done
fi
