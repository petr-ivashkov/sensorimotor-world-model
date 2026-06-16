# Planning-horizon sweep (Fig. 9)

Planning success vs. goal offset, using `train` seed 0:

```text
4 environments x 6 goal offsets x 4 methods = 96 jobs
```

Goal offsets are paired elementwise with the evaluation budget (`budget = 2 x offset`):

```text
25 -> 50   40 -> 80   55 -> 110   70 -> 140   85 -> 170   100 -> 200
```

Each job evaluates 100 tasks; for a fixed environment and offset all methods share the same
start–goal pairs. Learned methods load `experiments/train/results/<run_name>/` (run the
`train` stage first).

## Run

```bash
cd experiments/horizon_sweep
python generate_configs.py

# one job at a time: run.sh <env> <method> <goal_offset>
./run.sh tworoom inverse 70
./run.sh ogbcube random 100

python aggregate_results.py                # writes aggregated_results.csv
```

Per-run outputs land in
`experiments/horizon_sweep/results/<env>/<method>/offset_<goal_offset>/metrics.json`.

## Figure

```bash
jupyter notebook plot_results.ipynb
```

`plot_results.ipynb` loads `aggregated_results.csv` and displays the success-vs-offset figure
inline.
