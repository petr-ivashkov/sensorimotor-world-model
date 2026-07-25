# PLDM baseline

Five-seed evaluation of the official PLDM implementation under the shared planning
protocol:

```text
4 environments x 5 seeds = 20 training runs + 20 planning-evaluation jobs
```

## Method

The model and loss code are vendored byte-for-byte from the official
`stable-worldmodel` PLDM port pinned in `SOURCE.md`. Its architecture,
optimization, and objective defaults are preserved, except for matched context:

- ViT-Tiny encoder trained from scratch;
- predictor history `H=1`, matched to every learned method in the main figure;
- 100 epochs, batch size 128, AdamW with learning rate `5e-5`;
- objective:
  `prediction + 18 std + 0.7 std_t + 12 cov + 0.2 temporal alignment`.

The official default sets the IDM coefficient to `0`; this experiment leaves it at `0`.
It also leaves covariance-over-time, SIGReg, and temporal straightening inactive.
PLDM ships with `H=3`; only the history is changed to `H=1` for the controlled
main-figure comparison.

The only adaptations are experimental controls:

- every method receives the same full HDF5 training split and held-out validation split;
- planning uses the same CEM solver, 100 tasks, goal offset 25, budget 50, and task/policy
  seeds as `planning_eval`;
- checkpoints use the repository's strict-load format.

Outputs remain under this folder:

```text
results/train/<run_name>/
results/eval/<environment>/pldm/seed_<seed>/
```

## Cluster

From the planning repository root:

```bash
cd experiments/pldm
python generate_configs.py
mkdir -p logs

condor_submit_bid 100 train.sub

# Run only after training finishes; this must print 20/20.
python check_training.py

condor_submit_bid 100 eval.sub

python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```

Lightning output is written to `logs/condor.train.*.err`.

Single-run smoke test:

```bash
cd experiments/pldm
python generate_configs.py
RUNS_ROOT=$PWD/results_smoke/train ./train_run.sh pusht_pldm_seed0 \
  trainer.max_epochs=2 +trainer.limit_train_batches=1 \
  +trainer.limit_val_batches=1 loader.batch_size=2 \
  wandb.enabled=false artifacts.embedding_subset_size=4
```

## Local

The full sweep can be run sequentially with the same launchers:

```bash
cd planning/experiments/pldm
python generate_configs.py

while read -r run_name; do
  ./train_run.sh "$run_name" wandb.enabled=false
done < generated_configs/train_queue.txt

python check_training.py

while read -r run_name; do
  ./eval_run.sh "$run_name"
done < generated_configs/eval_queue.txt

python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```

## Transfer results from the cluster

Set the absolute checkout path on the cluster, then run from the local repository root.
This transfers analysis artifacts but not checkpoints or saved embeddings:

```bash
REMOTE_REPO=/absolute/path/to/cluster/checkout
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='metrics.json' \
  --include='metrics.csv' \
  --include='config.yaml' \
  --exclude='*' \
  "mpi-cluster:${REMOTE_REPO}/planning/experiments/pldm/results/" \
  planning/experiments/pldm/results/

cd planning/experiments/pldm
python generate_configs.py
python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```
