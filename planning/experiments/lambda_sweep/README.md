# Inverse-weight sweep

Five-seed sensitivity analysis for the inverse-dynamics weight:

```text
4 environments x 8 lambda values x 5 seeds = 160 train + 160 planning-eval runs
lambda in {0, 0.1, 0.3, 1, 3, 10, 30, 100}
```

Every run uses predictor history `H=1`. Planning uses 100 shared tasks per environment,
goal offset 25, and budget 50, matching the main planning evaluation. Training seeds vary
from 0 to 4; the sampled planning tasks are fixed within each environment while policy
seeds vary with the training seed.

The post-training diagnostics use 4,096 held-out sequences:

- `effective_rank`: effective rank of the centered validation embeddings;
- `mean_per_dim_variance`: mean population variance over latent dimensions;
- `mean_latent_vector_length`: mean Euclidean norm of a latent vector;
- `inverse_mse`: unweighted held-out inverse-model MSE;
- `forward_mse`: unweighted held-out one-step latent prediction MSE.

For `lambda=0`, `inverse_mse` measures the untrained inverse head and is included only as a
diagnostic. `train_run.sh` computes `diagnostics.json` immediately after each training run.

## Layout

```text
generate_configs.py       generated train/eval configs, manifest, and queue files
train_run.sh              train one model and collect held-out diagnostics
eval_run.sh               planning evaluation for one checkpoint
train.sub / eval.sub      HTCondor arrays over all 160 runs
aggregate_results.py      aggregated_results.csv plus results/summary_results.csv
plot_results.ipynb        planning success plus the five held-out diagnostics
```

Outputs are written under `results/train/<run_name>/` and
`results/eval/<run_name>/`.

## Cluster

From the planning repository root:

```bash
cd experiments/lambda_sweep
python generate_configs.py
mkdir -p logs

condor_submit_bid 100 train.sub

# Submit evaluation only after this prints 160/160 diagnostics.
python aggregate_results.py

condor_submit_bid 100 eval.sub

python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```

Lightning training output is written to `logs/condor.train.*.err`.

Single-run smoke test:

```bash
cd experiments/lambda_sweep
python generate_configs.py
RUNS_ROOT=$PWD/results_smoke/train ./train_run.sh ogbcube_lambda_10_seed0 \
  trainer.max_epochs=2 +trainer.limit_train_batches=5 \
  trainer.val_check_interval=5 trainer.limit_val_batches=2 \
  wandb.enabled=false
```

## Local

The same launchers work locally. A full sequential run is:

```bash
cd planning/experiments/lambda_sweep
python generate_configs.py

while read -r run_name; do
  ./train_run.sh "$run_name" wandb.enabled=false
done < generated_configs/train_queue.txt

while read -r run_name; do
  ./eval_run.sh "$run_name"
done < generated_configs/eval_queue.txt

python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```

## Transfer results from the cluster

Run this from the local repository root after setting the absolute checkout path on the
cluster. It transfers only analysis artifacts, not checkpoints.

```bash
REMOTE_REPO=/absolute/path/to/cluster/checkout
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='metrics.json' \
  --include='diagnostics.json' \
  --include='metrics.csv' \
  --exclude='*' \
  "mpi-cluster:${REMOTE_REPO}/planning/experiments/lambda_sweep/results/" \
  planning/experiments/lambda_sweep/results/

cd planning/experiments/lambda_sweep
python generate_configs.py
python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```
