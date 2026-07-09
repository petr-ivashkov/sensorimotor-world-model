#!/bin/bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/../.venv/bin/activate"

python -u "$EXPERIMENT_DIR/aggregate_results.py" "$@"
