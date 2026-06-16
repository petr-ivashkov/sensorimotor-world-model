#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"

export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [lambda_inv] [lambda_label]" >&2
    exit 2
fi

LAMBDA_INV="${1:-${LAMBDA_INV:-10}}"
LAMBDA_LABEL="${2:-${LAMBDA_LABEL:-lambda_${LAMBDA_INV//./p}}}"

case "$LAMBDA_LABEL" in
    lambda_*) ;;
    *) LAMBDA_LABEL="lambda_$LAMBDA_LABEL" ;;
esac

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

RUN_DIR="$RUNS_ROOT/$LAMBDA_LABEL"
if [ -e "$RUN_DIR" ] && [ "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" != "" ] && [ "${LWM_ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "Refusing to overwrite non-empty run directory: $RUN_DIR" >&2
    echo "Set LWM_ALLOW_OVERWRITE=1 to intentionally rerun this lambda." >&2
    exit 1
fi

CONFIG_ARGS=()
case " $* " in
    *" --config-path "*|*" --config-dir "*) ;;
    *) CONFIG_ARGS+=(--config-path "$EXPERIMENT_DIR") ;;
esac

case " $* " in
    *" --config-name "*) ;;
    *) CONFIG_ARGS+=(--config-name config) ;;
esac

echo "Launching TwoRoom inverse lambda sweep run: loss.inverse.weight=$LAMBDA_INV subdir=$LAMBDA_LABEL"
python -u "$REPO_ROOT/train.py" \
    "${CONFIG_ARGS[@]}" \
    "loss.inverse.weight=$LAMBDA_INV" \
    "subdir=$LAMBDA_LABEL" \
    "wandb.config.name=lambda_sweep_train_tworoom_${LAMBDA_LABEL}"
