# DINO-WM baseline (no proprioception)

Five-seed evaluation of DINO-WM under the shared planning protocol:

```text
4 environments x 5 seeds = 20 training runs + 20 planning-evaluation jobs
```

## Method

The experiment uses the DINO-WM implementation in the locked
`stable-worldmodel==0.0.6` dependency. Source details are in `SOURCE.md`.

- frozen pretrained DINOv2-Small patch encoder;
- DINO-WM causal predictor and action encoder;
- matched predictor history `H=1`;
- pure teacher-forcing MSE on the predicted non-action embedding;
- 10 epochs, effective batch size 256, AdamW with learning rate `1e-4` and
  weight decay `1e-3`;
- physical batch size 32 with eight-step mean-gradient accumulation, the shared
  optimizer-update-indexed linear-warmup cosine schedule, and `bf16` precision;
- pixels and actions only in every environment.

The model and planner never use or encode `proprio`, `observation`, agent
position, or any other simulator state. Simulator state is used only by the
common evaluation harness to initialize tasks and score success, as it is for
every method. Planning cost is computed only between predicted DINO patch
features and DINO patch features of the goal image.

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

The corrected runs use versioned names and directories so that checkpoints and
metrics from the earlier state-conditioned protocol cannot be mixed in:

```text
results/train/<environment>_dino_wm_noprop_seed<seed>/
results/eval/<environment>/dino_wm_noprop/seed_<seed>/
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

Before launching the replacement runs, remove only the incompatible
state-conditioned DINO-WM jobs and outputs:

```bash
cd experiments/dino_wm

condor_q "$USER" -autoformat ClusterId ProcId Args \
  | awk '$3 ~ /_dino_wm_seed[0-9]+$/ {print $1 "." $2}' \
  | xargs -r condor_rm

rm -rf results/train/{tworoom,reacher,pusht,ogbcube}_dino_wm_seed{0,1,2,3,4}
rm -rf results/eval/{tworoom,reacher,pusht,ogbcube}/dino_wm
rm -f aggregated_results.csv dino_wm_summary_results.csv
rm -f comparison_results.csv comparison_summary_results.csv
rm -f dino_wm_comparison.pdf
mkdir -p logs
find logs -maxdepth 1 -type f -name '*_dino_wm_seed*' -delete
```

Single-run smoke test:

```bash
cd experiments/dino_wm
RUNS_ROOT=$PWD/results_smoke/train ./train_run.sh pusht_dino_wm_noprop_seed0 \
  trainer.max_epochs=1 +trainer.limit_train_batches=2 \
  trainer.val_check_interval=1 \
  trainer.limit_val_batches=1 trainer.log_every_n_steps=1 \
  trainer.accumulate_grad_batches=1 loader.batch_size=2 \
  optimization_matching.enabled=false wandb.enabled=false
```

Do not evaluate checkpoints whose saved config lacks protocol
`matched_batch256_pixels_actions_v3`. Training, checkpoint validation,
evaluation, aggregation, and plotting all reject incompatible protocols.

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

Run from the local repository root after evaluation. This transfers metrics, not
checkpoints:

```bash
rsync -av \
  mpi-cluster:~/projects/sensorimotor-world-model/planning/experiments/dino_wm/results/eval/ \
  planning/experiments/dino_wm/results/eval/

cd planning/experiments/dino_wm
python generate_configs.py
python aggregate_results.py
jupyter nbconvert --to notebook --execute --inplace plot_results.ipynb
```
