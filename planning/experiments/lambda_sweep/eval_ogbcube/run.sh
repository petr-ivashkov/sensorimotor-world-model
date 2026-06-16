#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"

export PATH="/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

cd "$EXPERIMENT_DIR"
source "$REPO_ROOT/../.venv/bin/activate"

export REPO_ROOT
export EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-$REPO_ROOT/data/external}"
export GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-$REPO_ROOT/data/generated}"
export RUNS_ROOT="${RUNS_ROOT:-$EXPERIMENT_DIR/results}"
export STABLEWM_HOME="${STABLEWM_HOME:-$EXTERNAL_DATA_ROOT}"

/bin/mkdir -p "$EXTERNAL_DATA_ROOT" "$GENERATED_DATA_ROOT" "$RUNS_ROOT"

if [ "$#" -ne 0 ] && [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 [run_name] [goal_offset_steps]" >&2
    exit 2
fi

SELECTED_RUN="${1:-}"
SELECTED_GOAL_OFFSET="${2:-25}"
CONFIG_ROOT="$RUNS_ROOT/generated_configs"
MANIFEST_TAG="${SELECTED_RUN:-all}_${SELECTED_GOAL_OFFSET:-all}"
MANIFEST="$CONFIG_ROOT/manifest_${MANIFEST_TAG}.tsv"
/bin/mkdir -p "$CONFIG_ROOT"

python - "$EXPERIMENT_DIR/config.yaml" "$CONFIG_ROOT" "$MANIFEST" "$SELECTED_RUN" "$SELECTED_GOAL_OFFSET" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

config_path = Path(sys.argv[1])
config_root = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
selected_run = sys.argv[4]
selected_goal_offset = sys.argv[5]

base = OmegaConf.load(config_path)
goal_offsets = [int(x) for x in base.get("sweep", {}).get("goal_offset_steps", [])]
result_suffix = str(base.get("sweep", {}).get("result_suffix", ""))
if not goal_offsets:
    raise SystemExit("config.yaml must define sweep.goal_offset_steps")

if selected_goal_offset:
    selected_goal_offset = int(selected_goal_offset)
    if selected_goal_offset not in goal_offsets:
        raise SystemExit(
            f"goal_offset_steps={selected_goal_offset} is not in sweep.goal_offset_steps"
        )

rows = []
for run in base.runs:
    run_name = str(run.name)
    if selected_run and run_name != selected_run:
        continue
    for goal_offset in goal_offsets:
        if selected_goal_offset and goal_offset != selected_goal_offset:
            continue
        eval_budget = int(base.get("eval", {}).get("eval_budget", 2 * goal_offset))
        goal_tag = f"{goal_offset:03d}"
        result_tag = f"goal_offset_{goal_tag}{result_suffix}"

        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        cfg.runs = [OmegaConf.to_container(run, resolve=False)]
        cfg.eval.goal_offset_steps = goal_offset
        cfg.eval.eval_budget = eval_budget
        cfg.sweep.selected_run = run_name
        cfg.sweep.selected_goal_offset_steps = goal_offset
        cfg.sweep.selected_eval_budget = eval_budget
        cfg.sweep.selected_result_suffix = result_suffix
        cfg.sweep.selected_result_tag = result_tag

        cfg_dir = config_root / run_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{result_tag}.yaml"
        OmegaConf.save(config=cfg, f=cfg_path)
        rows.append((run_name, str(goal_offset), goal_tag, str(eval_budget), result_tag, str(cfg_path)))

if not rows:
    raise SystemExit(
        f"No sweep entries matched run_name={selected_run!r} "
        f"goal_offset_steps={selected_goal_offset!r}"
    )

manifest_path.write_text("\n".join("\t".join(row) for row in rows) + "\n")
print(f"Wrote {len(rows)} generated configs to {config_root}")
PY

while IFS=$'\t' read -r RUN_NAME GOAL_OFFSET GOAL_TAG EVAL_BUDGET RESULT_TAG CONFIG_PATH; do
    COMBO_ROOT="$RUNS_ROOT/$RUN_NAME/$RESULT_TAG"
    /bin/mkdir -p "$COMBO_ROOT"
    echo "Evaluating $RUN_NAME at goal_offset_steps=$GOAL_OFFSET eval_budget=$EVAL_BUDGET"
    if [ "${LWM_DRY_RUN:-0}" = "1" ]; then
        echo "Dry run: RUNS_ROOT=$COMBO_ROOT python -u $REPO_ROOT/eval.py --config $CONFIG_PATH"
        continue
    fi
    RUNS_ROOT="$COMBO_ROOT" python -u "$REPO_ROOT/eval.py" --config "$CONFIG_PATH"
done < "$MANIFEST"
