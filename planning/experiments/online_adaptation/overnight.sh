#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
PYTHON_BIN="${SMWM_PYTHON:-$RESEARCH_ROOT/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/smwm-matplotlib-${USER}"
export PYTHONPATH="$RESEARCH_ROOT:$RESEARCH_ROOT/planning${PYTHONPATH:+:$PYTHONPATH}"
OUTPUT_ROOT=""
previous=""
for arg in "$@"; do
    if [[ "$previous" == "--output-root" ]]; then OUTPUT_ROOT="$arg"; fi
    previous="$arg"
done
export_review() {
    local result=$?
    if [[ -n "$OUTPUT_ROOT" && -d "$OUTPUT_ROOT" ]]; then
        "$PYTHON_BIN" -B -m planning.experiments.online_adaptation.export_review "$OUTPUT_ROOT" || true
    fi
    exit "$result"
}
trap export_review EXIT
bash "$EXPERIMENT_DIR/run.sh" "$@"
