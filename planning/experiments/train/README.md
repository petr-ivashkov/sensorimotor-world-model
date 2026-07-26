# Training

Trains the world model for every environment, method, and seed used in the paper:

```text
4 environments x 3 methods x 5 seeds = 60 runs
```

- **Environments:** TwoRoom, Reacher, Push-T, OGBench-Cube
- **Methods:**
  - `inverse` (ours): forward prediction + inverse dynamics, SIGReg off
  - `forward_only`: forward prediction only (`loss.inverse.weight=0`, `loss.sigreg.weight=0`)
  - `sigreg`: forward prediction + SIGReg (`loss.sigreg.weight=0.09`), inverse off

Configs inherit from `config/train/base.yaml` and `config/train/data/<env>.yaml`; only the
method weights, seed, run name, and one-frame settings (`wm.history_size=1`,
`data.dataset.num_steps=2`) are varied.

The inverse-dynamics weight is fixed across environments:

```text
TwoRoom: 10   Reacher: 10   Push-T: 10   OGBench-Cube: 10
```

## Run

```bash
# 1. Generate the per-run configs + manifest
cd experiments/train
python generate_configs.py        # writes configs, manifest.tsv, and train_queue.txt

# 2. Train one run
./run.sh tworoom_inverse_lambda_10_seed0

# Or submit all 60 runs
condor_submit_bid 100 train.sub

# 3. Require all 60 matched runs before evaluation
python check_training.py
```

`run.sh <run_name>` activates the project venv, resolves data roots, and launches `train.py`
with the matching generated config. Each run writes to
`experiments/train/results/<run_name>/` (checkpoint, resolved `config.yaml`, embeddings).
These outputs are consumed by the `planning_eval`, `horizon_sweep`, and `embedding_analysis`
stages.
