# Combined IDR + SIGReg

Controlled test of whether inverse-dynamics regularization complements SIGReg:

```text
4 environments x 5 seeds = 20 training runs + 20 planning-evaluation jobs
```

- **Objective:** forward loss + `10 × inverse loss` + `0.09 × SIGReg loss`
- **History:** `H=1`
- **Environments:** TwoRoom, Reacher, Push-T, OGBench-Cube
- **Seeds:** 0–4

All other training settings are inherited from the standard train configs. Planning uses
100 tasks, goal offset 25, and budget 50. Its task and policy seeds exactly match
`planning_eval`, so results are paired with the existing methods by environment and seed.
The `H=1` IDR and SIGReg runs in the history ablation are the exact single-regularizer
controls for this combined objective.

Outputs stay under this folder:

```text
results/train/<run_name>/
results/eval/<environment>/idr_sigreg/seed_<seed>/
```

## Cluster

From the planning repository root:

```bash
cd experiments/idr_sigreg
python generate_configs.py
mkdir -p logs

condor_submit_bid 100 train.sub

# Run only after training has finished; this must print 20/20.
python check_training.py

condor_submit_bid 100 eval.sub

python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```

Lightning training output is written to `logs/condor.train.*.err`.
Aggregation writes the 20-run table and combined-only mean/SEM table; the notebook writes
the exact-control comparison table and figure.

Single-run smoke test:

```bash
cd experiments/idr_sigreg
python generate_configs.py
RUNS_ROOT=$PWD/results_smoke/train ./train_run.sh ogbcube_idr_sigreg_seed0 \
  trainer.max_epochs=2 +trainer.limit_train_batches=5 \
  trainer.val_check_interval=5 trainer.limit_val_batches=2 \
  wandb.enabled=false
```

## Local

```bash
cd planning/experiments/idr_sigreg
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

## Transfer analysis results from the cluster

Set the absolute checkout path on the cluster, then run from the local repository root:

```bash
REMOTE_REPO=/absolute/path/to/cluster/checkout
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='metrics.json' \
  --include='metrics.csv' \
  --exclude='*' \
  "mpi-cluster:${REMOTE_REPO}/planning/experiments/idr_sigreg/results/" \
  planning/experiments/idr_sigreg/results/

cd planning/experiments/idr_sigreg
python generate_configs.py
python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```
