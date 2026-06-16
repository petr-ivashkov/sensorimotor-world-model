#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <env> <method> <goal_offset>" >&2
    echo "  env: tworoom | reacher | pusht | ogbcube" >&2
    echo "  method: forward_only | inverse | sigreg | random" >&2
    echo "  goal_offset: 25 | 40 | 55 | 70 | 85 | 100" >&2
    exit 2
fi

ENV_SLUG="$1"
METHOD="$2"
GOAL_OFFSET="$3"

case "$ENV_SLUG" in
    tworoom|reacher|pusht|ogbcube) ;;
    *)
        echo "Unsupported env=$ENV_SLUG" >&2
        exit 2
        ;;
esac

case "$METHOD" in
    forward_only|inverse|sigreg|random) ;;
    *)
        echo "Unsupported method=$METHOD" >&2
        exit 2
        ;;
esac

case "$GOAL_OFFSET" in
    25|40|55|70|85|100) ;;
    *)
        echo "Unsupported goal_offset=$GOAL_OFFSET" >&2
        exit 2
        ;;
esac

RUN_LABEL="offset_${GOAL_OFFSET}"
JOB_NAME="${ENV_SLUG}_${METHOD}_${RUN_LABEL}"

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"
export BASE_RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$BASE_RUNS_ROOT"

CONFIG_ROOT="$EXPERIMENT_DIR/generated_configs"
CONFIG_FILE="$CONFIG_ROOT/$JOB_NAME.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    python "$EXPERIMENT_DIR/generate_configs.py" --output-dir "$CONFIG_ROOT"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Generated config does not exist: $CONFIG_FILE" >&2
    exit 1
fi

DATASET_FILE="$(python - "$CONFIG_ROOT/manifest.tsv" "$JOB_NAME" <<'PY'
import csv
import sys

manifest_path, job_name = sys.argv[1:3]
with open(manifest_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["job_name"] == job_name:
            print(row["dataset_file"])
            raise SystemExit(0)
raise SystemExit(f"Job {job_name!r} not found in {manifest_path}")
PY
)"

if [ "${LWM_DRY_RUN:-0}" != "1" ] && [ ! -f "$EXTERNAL_DATA_ROOT/$DATASET_FILE" ]; then
    echo "Evaluation split $DATASET_FILE missing under $EXTERNAL_DATA_ROOT; generating episode splits..."
    python -u "$REPO_ROOT/scripts/make_episode_splits.py" --root "$EXTERNAL_DATA_ROOT"
fi

METHOD_ROOT="$BASE_RUNS_ROOT/$ENV_SLUG/$METHOD"
/bin/mkdir -p "$METHOD_ROOT"

echo "Evaluating $JOB_NAME"
echo "  config: $CONFIG_FILE"
echo "  output: $METHOD_ROOT/$RUN_LABEL"

if [ "${LWM_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run: RUNS_ROOT=$METHOD_ROOT python -u $REPO_ROOT/eval.py --config $CONFIG_FILE"
    exit 0
fi

RUNS_ROOT="$METHOD_ROOT" python -u "$REPO_ROOT/eval.py" --config "$CONFIG_FILE"
