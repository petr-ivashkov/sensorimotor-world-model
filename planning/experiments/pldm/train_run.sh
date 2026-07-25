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
    echo "Usage: $0 <run_name> [extra train.py overrides...]" >&2
    exit 2
fi

RUN_NAME="$1"
shift

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results/train}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

CONFIG_ROOT="$EXPERIMENT_DIR/generated_configs/train"
CONFIG_FILE="$CONFIG_ROOT/$RUN_NAME.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    python "$EXPERIMENT_DIR/generate_configs.py"
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Generated config does not exist: $CONFIG_FILE" >&2
    exit 1
fi

RUN_DIR="$RUNS_ROOT/$RUN_NAME"
RUN_DIR_NONEMPTY="$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)"
if [ -e "$RUN_DIR" ] \
    && [ "$RUN_DIR_NONEMPTY" != "" ] \
    && [ "${LWM_ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "Refusing to overwrite non-empty run directory: $RUN_DIR" >&2
    echo "Set LWM_ALLOW_OVERWRITE=1 to intentionally rerun this job." >&2
    exit 1
fi

echo "Launching PLDM training run: $RUN_NAME"
python -u "$EXPERIMENT_DIR/train.py" \
    --config-path "$CONFIG_ROOT" \
    --config-name "$RUN_NAME" \
    "$@"
