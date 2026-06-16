# Planning evaluation (Fig. 5)

Goal-conditioned planning success for the trained models:

```text
4 environments x 4 methods x 5 seeds/repeats = 80 jobs
```

- **Methods:** `inverse` (ours), `forward_only`, `sigreg`, `random`
- **Fixed settings:** `num_eval=100`, `goal_offset_steps=25`, `eval_budget=50`

Learned methods load checkpoints from `experiments/train/results/<run_name>/` (run the
`train` stage first). Random jobs set `policy: random` and load no checkpoint. For each
environment, every method/seed evaluates on the same 100 start–goal pairs.

## Run

```bash
cd experiments/planning_eval
python generate_configs.py                 # needs experiments/train/.../manifest.tsv

# one job at a time: run.sh <env> <method> <seed_or_repeat>
./run.sh tworoom inverse 0
./run.sh pusht random 3

# aggregate per-run metrics.json into a single table
python aggregate_results.py                # writes aggregated_results.csv
```

Per-run outputs land in `experiments/planning_eval/results/<env>/<method>/<run_label>/metrics.json`.

## Figure

```bash
jupyter notebook plot_results.ipynb
```

`plot_results.ipynb` loads `aggregated_results.csv` and displays the planning-success figure
(mean ± SE over seeds/repeats) inline.
