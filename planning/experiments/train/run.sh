#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

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
export RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

CONFIG_ROOT="$EXPERIMENT_DIR/generated_configs"
CONFIG_FILE="$CONFIG_ROOT/$RUN_NAME.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    python "$EXPERIMENT_DIR/generate_configs.py" --output-dir "$CONFIG_ROOT"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Generated config does not exist: $CONFIG_FILE" >&2
    exit 1
fi

RUN_DIR="$RUNS_ROOT/$RUN_NAME"
if [ "${LWM_FRESH_START:-0}" = "1" ] && [ -e "$RUN_DIR" ]; then
    BACKUP_ROOT="$RUNS_ROOT/_rerun_backups"
    BACKUP_DIR="$BACKUP_ROOT/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
    /bin/mkdir -p "$BACKUP_ROOT"
    echo "LWM_FRESH_START=1: moving existing run directory to $BACKUP_DIR"
    /bin/mv "$RUN_DIR" "$BACKUP_DIR"
fi

if [ -e "$RUN_DIR" ] && [ "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" != "" ] && [ "${LWM_ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "Refusing to overwrite non-empty run directory: $RUN_DIR" >&2
    echo "Set LWM_ALLOW_OVERWRITE=1 to intentionally rerun this final-training job." >&2
    exit 1
fi

echo "Launching final training run: $RUN_NAME"
python -u "$REPO_ROOT/train.py" \
    --config-path "$CONFIG_ROOT" \
    --config-name "$RUN_NAME" \
    "$@"
