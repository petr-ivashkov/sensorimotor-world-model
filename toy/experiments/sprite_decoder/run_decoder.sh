#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/../.venv/bin/activate"

# Require a CUDA GPU; abort early if none is available.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    host="${HOSTNAME:-$(cat /proc/sys/kernel/hostname 2>/dev/null || echo unknown)}"
    echo "ERROR: CUDA unavailable on ${host}; this experiment requires a GPU. Aborting." >&2
    exit 1
fi

CONFIG_FILE="${1:-config_decoder_xyt.yaml}"
python -u "$REPO_ROOT/train_decoder.py" --config "$EXPERIMENT_DIR/$CONFIG_FILE" "${@:2}"
