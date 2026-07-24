#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

if [ "${LWM_SKIP_CUDA_MODULE:-0}" = "1" ]; then
    echo "Skipping module load because LWM_SKIP_CUDA_MODULE=1"
elif command -v module >/dev/null 2>&1; then
    module load cuda
else
    echo "module command not available; continuing without module load"
fi

export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <run_name> [extra eval.py overrides...]" >&2
    exit 2
fi

RUN_NAME="$1"
shift

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"
export RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results/eval}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

CONFIG_FILE="$EXPERIMENT_DIR/generated_configs/eval/$RUN_NAME.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    python "$EXPERIMENT_DIR/generate_configs.py"
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Generated config does not exist: $CONFIG_FILE" >&2
    exit 1
fi

echo "Evaluating history-ablation run: $RUN_NAME"
python -u "$REPO_ROOT/eval.py" --config "$CONFIG_FILE" "$@"
