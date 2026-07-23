#!/bin/bash
set -euo pipefail

# Like run.sh, but takes an explicit config instead of the <env> <method> <seed> triple.

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

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"

CONFIG_FILE="${1:-$EXPERIMENT_DIR/smoke_h3_config.yaml}"
shift || true

CKPT_DIR="$REPO_ROOT/experiments/train/results_smoke/smoke_h3"
if [ ! -f "$CKPT_DIR/checkpoints/last.ckpt" ]; then
    echo "ERROR: $CKPT_DIR/checkpoints/last.ckpt not found." >&2
    echo "Run experiments/train/smoke_h3.sub first." >&2
    exit 1
fi

OUT_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results_smoke}"
/bin/mkdir -p "$OUT_ROOT"

echo "Evaluating history_size=3 smoke checkpoint"
echo "  config: $CONFIG_FILE"
echo "  ckpt:   $CKPT_DIR"
echo "  output: $OUT_ROOT"

RUNS_ROOT="$OUT_ROOT" python -u "$REPO_ROOT/eval.py" --config "$CONFIG_FILE" "$@"
