#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [env_name] [method]" >&2
    exit 2
fi

SELECTED_ENV="${1:-}"
SELECTED_METHOD="${2:-}"

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/outputs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$EXPERIMENT_DIR/.mplconfig}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT" "$MPLCONFIGDIR"

ARGS=(--config "$EXPERIMENT_DIR/config.yaml" --output-root "$RUNS_ROOT")
if [ -n "$SELECTED_ENV" ]; then
    ARGS+=(--env "$SELECTED_ENV")
fi
if [ -n "$SELECTED_METHOD" ]; then
    ARGS+=(--method "$SELECTED_METHOD")
fi
if [ "${LWM_EXPORT_OVERWRITE:-0}" = "1" ]; then
    ARGS+=(--overwrite)
fi

python -u "$EXPERIMENT_DIR/export_embeddings.py" "${ARGS[@]}"
