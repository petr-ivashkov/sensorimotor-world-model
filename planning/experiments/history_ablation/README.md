# History-length ablation (IDR vs SIGReg)

Controlled comparison of **IDR** and **SIGReg** across forward-model history lengths, to
answer the reviewers' question of whether SIGReg's history length confounds the main result.

```text
4 environments x 2 methods x 3 histories x 5 seeds = 120 training runs (and 120 eval jobs)
```

- **Environments:** TwoRoom, Reacher, Push-T, OGBench-Cube
- **Methods:** `idr` (inverse-dynamics reg., fixed **λ = 10**), `sigreg` (weight 0.09)
- **Histories:** `wm.history_size ∈ {1, 2, 3}` (`num_steps` set to `H+1` automatically)
- **Seeds:** 0–4

Everything else — encoder, predictor, optimizer, datasets, 10-epoch budget, CEM/MPC eval —
is inherited unchanged from `/train/base`, `/train/data/<env>`, `/eval/base`, and
`/eval/env/<env>`. The `run.sh` wrappers call the repo-root `train.py` / `eval.py`; no
training or evaluation logic is duplicated here. Eval reuses `planning_eval`'s per-environment
task seeds, so these numbers are directly comparable to the main Fig. 5 results, and history is
read from each checkpoint's saved config (eval configs are history-agnostic).

## Layout

```text
generate_configs.py      # writes generated_configs/{train,eval}/*.yaml + manifest.tsv + queue files
train_run.sh <run_name>  # activate env, launch train.py for one run
eval_run.sh  <run_name>  # activate env, launch eval.py  for one checkpoint
train.sub  eval.sub      # HTCondor: queue over generated_configs/{train,eval}_queue.txt
aggregate_results.py     # eval metrics.json -> aggregated_results.csv
diagnostics.ipynb        # per-run training curves + completion / collapse checks
plot_results.ipynb       # grouped success bars + final mean±SEM table
```

Outputs land in `results/train/<run_name>/` and `results/eval/<run_name>/` (both gitignored).

## Run (cluster)

```bash
cd experiments/history_ablation
python generate_configs.py            # 120 train + 120 eval configs, manifest, queue files
mkdir -p logs

# 1. train (120 jobs, ~4 GPU-h each)
condor_submit_bid 100 train.sub

# 2. after training finishes, verify:
jupyter nbconvert --to notebook --execute --inplace diagnostics.ipynb   # expect "Complete: 120/120"

# 3. plan (120 jobs) — reads the trained checkpoints
condor_submit_bid 100 eval.sub

# 4. aggregate + plot
python aggregate_results.py           # -> aggregated_results.csv
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb  # figure + history_ablation_table.csv
```

Single-run smoke test before the full sweep:

```bash
RUNS_ROOT=$PWD/results/train ./train_run.sh ogbcube_idr_h3_seed0 \
  trainer.max_epochs=1 +trainer.limit_train_batches=5 trainer.val_check_interval=5 \
  trainer.limit_val_batches=2 wandb.enabled=false
```

`diagnostics.ipynb` gates `eval.sub`: don't submit planning until every training run shows
`ok` and no collapse is flagged.
