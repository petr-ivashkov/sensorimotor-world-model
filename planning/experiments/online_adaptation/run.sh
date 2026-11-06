#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
PYTHON_BIN="${SMWM_PYTHON:-$RESEARCH_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Set SMWM_PYTHON to the existing cluster checkout's .venv/bin/python." >&2
    exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export REPO_ROOT="$RESEARCH_ROOT/planning"
exec "$PYTHON_BIN" -B -u "$EXPERIMENT_DIR/run.py" "$@"
