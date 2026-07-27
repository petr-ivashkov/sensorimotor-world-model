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
export BASE_RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results/eval}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$BASE_RUNS_ROOT"

CONFIG_ROOT="$EXPERIMENT_DIR/generated_configs/eval"
CONFIG_FILE="$CONFIG_ROOT/$RUN_NAME.yaml"
MANIFEST="$EXPERIMENT_DIR/generated_configs/manifest.tsv"
if [ ! -f "$CONFIG_FILE" ] || [ ! -f "$MANIFEST" ]; then
    python "$EXPERIMENT_DIR/generate_configs.py"
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Generated config does not exist: $CONFIG_FILE" >&2
    exit 1
fi

IFS=$'\t' read -r ENV_SLUG RUN_LABEL DATASET_FILE RESULT_GROUP < <(
    python - "$MANIFEST" "$RUN_NAME" <<'PY'
import csv
import sys

manifest_path, run_name = sys.argv[1:3]
with open(manifest_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row["run_name"] == run_name:
            print(
                row["env"],
                row["run_label"],
                row["dataset_file"],
                row["result_group"],
                sep="\t",
            )
            raise SystemExit(0)
raise SystemExit(f"Run {run_name!r} not found in {manifest_path}")
PY
)

if [ "${LWM_DRY_RUN:-0}" != "1" ] \
    && [ ! -f "$EXTERNAL_DATA_ROOT/$DATASET_FILE" ]; then
    echo "Evaluation split $DATASET_FILE missing; generating episode splits."
    python -u "$REPO_ROOT/scripts/make_episode_splits.py" \
        --root "$EXTERNAL_DATA_ROOT"
fi

METHOD_ROOT="$BASE_RUNS_ROOT/$ENV_SLUG/$RESULT_GROUP"
RUN_DIR="$METHOD_ROOT/$RUN_LABEL"
/bin/mkdir -p "$METHOD_ROOT"

if [ -f "$RUN_DIR/metrics.json" ] \
    && [ "${LWM_ALLOW_OVERWRITE:-0}" != "1" ]; then
    echo "Refusing to overwrite completed evaluation: $RUN_DIR/metrics.json" >&2
    echo "Set LWM_ALLOW_OVERWRITE=1 to intentionally rerun this job." >&2
    exit 1
fi

echo "Evaluating DINO-WM run: $RUN_NAME"
if [ "${LWM_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run: RUNS_ROOT=$METHOD_ROOT"
    echo "python -u $EXPERIMENT_DIR/eval.py --config $CONFIG_FILE $*"
    exit 0
fi

RUNS_ROOT="$METHOD_ROOT" python -u "$EXPERIMENT_DIR/eval.py" \
    --config "$CONFIG_FILE" \
    "$@"
