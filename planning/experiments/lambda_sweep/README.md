# Inverse-weight (lambda) sweep (Fig. 7, Table 1)

Sensitivity of planning success to the inverse-dynamics weight `lambda`, per environment.
For each environment we train one model per `lambda` and then run goal-conditioned planning
evaluation, sweeping:

```text
lambda in {0, 0.1, 0.3, 1, 3, 10, 30, 100}
```

Layout (one pair of folders per environment):

```text
train_<env>/   trains the 8 lambda models           -> results/lambda_<value>/
eval_<env>/    planning eval for those checkpoints   -> results/lambda_<value>/...
plot_results.ipynb   combines all four environments  -> Fig. 7
```

## Train the sweep

`train_<env>/run.sh <lambda>` overrides only `loss.inverse.weight` (and the run subdir/label):

```bash
cd experiments/lambda_sweep/train_pusht
for l in 0 0.1 0.3 1 3 10 30 100; do ./run.sh "$l"; done
```

## Evaluate the sweep

`eval_<env>/run.sh` reads the matching `train_<env>/results/lambda_*` checkpoints. With no
arguments it evaluates every lambda at every offset in the config:

```bash
cd experiments/lambda_sweep/eval_pusht
./run.sh
```

Repeat both steps for `tworoom`, `reacher`, and `ogbcube`.

## Figure

```bash
cd experiments/lambda_sweep
jupyter notebook plot_results.ipynb
```

`plot_results.ipynb` reads the `eval_<env>/results/` metrics for all four environments and
displays the success-vs-lambda figure (Fig. 7) and the underlying numbers (Table 1) inline.
The red marker in each panel is the `lambda` chosen for the main experiments.