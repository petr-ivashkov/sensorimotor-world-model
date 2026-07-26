# DINO-WM baseline

Five-seed evaluation of DINO-WM under the shared planning protocol:

```text
4 environments x 5 seeds = 20 training runs + 20 planning-evaluation jobs
```

## Method

The experiment uses the DINO-WM implementation in the locked
`stable-worldmodel==0.0.6` dependency. Source details are in `SOURCE.md`.

- frozen pretrained DINOv2-Small patch encoder;
- DINO-WM causal predictor and action/state encoders;
- matched predictor history `H=1`;
- pure teacher-forcing MSE on the predicted non-action embedding;
- 10 epochs, effective batch size 256, AdamW with learning rate `1e-4` and
  weight decay `1e-3`;
- physical batch size 32 with eight-step mean-gradient accumulation, the shared
  optimizer-update-indexed linear-warmup cosine schedule, and `bf16` precision;
- state and action conditioning where supported, with action-only OGBench-Cube.

DINO-WM ships with `H=3`; only history is changed to `H=1` for the controlled
main-figure comparison. Its architecture and objective are unchanged; optimization
uses the shared main-figure protocol.

Every method receives the same full HDF5 training split and held-out validation
split. Planning uses the same CEM solver, 100 tasks, goal offset 25, budget 50,
task seeds, and policy seeds as `planning_eval`.

Each epoch is truncated to `8 * floor(N / 256)` micro-batches. After gradient
accumulation, this gives exactly `floor(N / 256)` optimizer updates, matching the
main methods without requiring a memory-heavy physical batch of 256. Each
micro-batch loss is divided by eight before backward, so one accumulated update
equals the mean loss over 256 examples. Gradient clipping and the learning-rate
scheduler run once per optimizer update.

Validation and logging intervals are scaled by eight so they occur after the
same number of optimizer updates and examples as in the main training pipeline.
The loader uses the shared 24 workers and prefetch factor 4.

Outputs remain under this folder:

```text
results/train/<run_name>/
results/eval/<environment>/dino_wm/seed_<seed>/
```

## Cluster

From the planning repository root:

```bash
cd experiments/dino_wm
python cache_backbone.py
python generate_configs.py
mkdir -p logs

condor_submit_bid 100 train.sub

# Run only after training finishes; this validates configs and checkpoint state.
python check_training.py

condor_submit_bid 100 eval.sub

python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```

`cache_backbone.py` downloads DINOv2-Small once into the shared Hugging Face
cache. Lightning output is written to `logs/condor.train.*.err`.

Single-run smoke test:

```bash
cd experiments/dino_wm
RUNS_ROOT=$PWD/results_smoke/train ./train_run.sh pusht_dino_wm_seed0 \
  trainer.max_epochs=1 +trainer.limit_train_batches=2 \
  trainer.val_check_interval=1 \
  trainer.limit_val_batches=1 trainer.log_every_n_steps=1 \
  trainer.accumulate_grad_batches=1 loader.batch_size=2 \
  optimization_matching.enabled=false wandb.enabled=false
```

Do not evaluate checkpoints whose saved config lacks protocol
`matched_batch256_mean_accumulation_v2`.

## Local

The full sweep can be run sequentially with the same launchers:

```bash
cd planning/experiments/dino_wm
python cache_backbone.py
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

Set the absolute checkout path on the cluster, then run from the local repository
root. This transfers metrics and configs, not checkpoints:

```bash
REMOTE_REPO=/absolute/path/to/cluster/checkout
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='metrics.json' \
  --include='metrics.csv' \
  --include='config.yaml' \
  --exclude='*' \
  "mpi-cluster:${REMOTE_REPO}/planning/experiments/dino_wm/results/" \
  planning/experiments/dino_wm/results/

cd planning/experiments/dino_wm
python generate_configs.py
python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```
