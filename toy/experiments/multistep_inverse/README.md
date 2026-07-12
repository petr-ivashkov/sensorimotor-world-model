# Multistep inverse dynamics (first-action) + oracle-subgoal goal reaching

Extends the joint objective `L = L_fwd + λ·L_inv` to the multi-step inverse
construction of Lamb et al. (arXiv 2207.08229, *Guaranteed Discovery of
Control-Endogenous Latent State with Multi-Step Inverse Models*),
implemented in `train_multistep.py`:

```
L = ||g_ϕ(z_t, a_t) − z_{t+1}||² + λ·||h_ψ(z_t, z_{t+k}, k) − a_t||²,
k ~ Uniform{k_min .. k_max}  per training example
```

The inverse head decodes the segment *endpoints* into the **first action**
`a_t` — not the sequence, not an aggregate. The horizon is an explicit input
(one-hot through the first layer, `models.MultistepInverseModel`): the
identifiability argument is per-horizon, and with i.i.d. uniform actions the
regression optimum is `E[a_t | endpoints, k] = S/k` (steps are exchangeable
given the endpoint displacement S), which a horizon-agnostic head cannot
represent for two different k at the same endpoints. The forward loss stays
single-step. `horizon_k_max: 1` reproduces `train.py` **bitwise** (same data
RNG sequence, same models via `train.build_models`, same update sequence —
asserted end-to-end in the tests).

**Flagged assumption (tested, not fixed).** Lamb et al.'s guarantee holds for
finite action spaces with deterministic endogenous dynamics. Ours is
continuous: `(z_t, z_{t+k})` does not determine `a_t` — many action sequences
share endpoints — so the MSE head regresses the conditional mean and its
residual grows with k. We evaluate the objective empirically as a
regularizer and instrument for the degradation: the eval pass logs the
inverse loss **per horizon k** (`train_history.pt: eval.inv_per_k`, `inv@k`
columns in `summary.csv`, a dedicated panel in `analyze.ipynb`), so larger-k
failure is visible rather than hidden in aggregates.

## Runs

5 environments × `k_max ∈ {1, 2, 4, 8}` (`k_min = 1` everywhere) —
`config_<env>_kmax<K>.yaml`:

| env | world | action_dim | question it answers |
|---|---|---|---|
| `single_dot` | 1 independent dot | 2 | base case |
| `2_independent` | 2 independent dots | 4 | multiple controllable DOFs |
| `1_coupled_pair` | 1 coupled pair | 2 | shared-action structure |
| `1_dot_1_random` | 1 dot + 1 random | 2 | is the distractor ignored? |
| `sprite_xyt` | full-control sprite (x, y, θ) | 3 | continuous pose, θ wrap |

Note: `sprite_xyt` keeps the matched-rim-speed `max_delta_theta` (~61°/step,
the `sprite_base` default) instead of `sprite_decoder/config_xyt`'s ±π: with
±π steps the endpoint θ already wraps at k=2, making the first action
unidentifiable immediately and masking the horizon effect under study.

## Goal-reaching evaluation (`eval_goal_reaching.py`)

Oracle subgoals — ground-truth intermediate states encoded with `f_θ` — are
decoded pairwise with `h_ψ` and executed in two modes (`eval.execution`):

- **replanning** — after every executed step, re-encode the current
  observation and decode toward the active subgoal; the queried horizon
  counts down from the subgoal's oracle gap to 1 (capped at the trained
  `k_max`). Subgoals advance on arrival or budget
  (`ceil(budget_factor·spacing) + budget_slack` steps).
- **open_loop** — decode each consecutive oracle pair once,
  `â_j = h_ψ(z_{g_j}, z_{g_{j+1}}, gap)`, execute the sequence blind. At
  spacing 1 this is full-sequence replay; at spacing m > 1 only each gap's
  first action is recoverable from endpoints, so open-loop is the diagnostic
  lower bound and replanning the practical mode.

Subgoals are deliberately oracle states: decoding is isolated from subgoal
generation, so failures are attributable. Success is scored on the
controllable DOFs only; RANDOM dots keep moving during execution and their
residual error is reported separately (should stay at chance).

## Runbook (cluster)

Everything is seeded from the configs: training `seed: 42` for **all** runs
(shared via `dot_base` / `sprite_base`, so runs are paired across `k_max` —
identical training states), eval `seed: 123` (shared via `goal_base`, so
episodes are paired across runs, spacings, and modes). No other seeds exist.

```bash
# 0) once per checkout — env + correctness gate (~3 min, CPU)
cd ~/projects/sensorimotor-world-model && uv sync
cd toy && python tests/test_multistep_inverse.py

# 1) stage 1 — frozen single-step reference (k_max=1 == train.py), 5 GPU jobs
cd experiments/multistep_inverse
condor_submit_bid 100 train_ref.sub

# 2) when stage 1 finishes: sanity-check it, then submit the sweep
#    (inspect condor.*.out: losses decreasing, 'horizon_k=U{1..1}')
condor_submit_bid 100 train_sweep.sub          # k_max in {2,4,8}, 15 GPU jobs

# 3) after ALL training jobs finish (eval reads results/<run>/model.pt):
condor_submit_bid 100 eval.sub                 # 5 CPU jobs, one per env

# 4) after eval finishes:
condor_submit_bid 100 aggregate.sub            # writes results/summary.csv
```

Dependencies: 2 needs 1 only for the sanity gate (the jobs are independent);
3 needs 1+2 complete for every run it evaluates (a missing run is reported
and skipped, so a partial eval is safe to rerun); 4 needs 3. Fetch results to
your machine and open the notebook:

```bash
rsync -avz --progress --exclude='model.pt' \
  mpi-cluster:~/projects/sensorimotor-world-model/toy/experiments/multistep_inverse/results/ \
  ~/Projects/sensorimotor-world-model/toy/experiments/multistep_inverse/results/
# then: open analyze.ipynb and Run All
```

Run directories keep the `train.py` layout
(`results/<run>/{model.pt, config.yaml, train_history.pt, embeddings.pt}`
plus `goal_reaching.pt`), so `train_decoder.py` and the existing notebooks
work on them unchanged. Results from the earlier mean-action version
(`<env>_k<k>` run names) are left alone: the notebook only loads `_kmax`
runs and `aggregate_results.py` skips their stale `goal_reaching.pt`.

**Resources.** Dot-world runs: 100 epochs; sprite: 200. Segments carry 3
frames per sample: ≈ 37 GB resident per training run at the default 250 k
samples, transiently ~2× while stacking (`request_memory = 122880` covers
it). Eval is CPU-bound rendering: minutes per dot-world run, ~10–20 min per
sprite run.

## Verify

`tests/test_multistep_inverse.py` (also runs under pytest) pins down, in
order: segment datasets at `k_min=k_max=1` are **bitwise identical** to the
transition datasets in both worlds; random-k segments are self-consistent
(k in range, deterministic per index, first action matches the first
transition, states in-canvas); the head is genuinely conditioned on k;
`train_multistep.compute_losses` at `k_max=1` equals `train.compute_losses`
exactly; a short `train_multistep.py` `k_max=1` run produces
**bitwise-identical weights** to `train.py`; the closed-loop steppers replay
recorded actions exactly; the goal-reaching eval runs end-to-end in both
modes (and the k-conditioned decode path for `k_max>1`); aggregation and the
shipped configs/submit files are consistent.

## Reading the results

- **Does the first-action objective still recover the controllable DOFs?**
  `summary.csv` / notebook Q1: `eff_rank` near `true_dim`, `r2_ctrl` near 1
  across `k_max`, `r2_unctrl` ≈ 0 (probe fit held-out on the eval split — an
  independent uniform sample of states); PCA knee at the true dim.
- **Does oracle decoding still reach goals as spacing grows?** Q2:
  `succ@m` (replanning) per `k_max`; `ol_succ@m` shows the open-loop
  collapse with spacing that replanning must fix. `inv@k` columns show the
  first-action regression degrading with horizon (the multimodality caveat
  made visible).
- **Does planning ignore the uncontrollable dot?** Q3: `1_dot_1_random`
  tracks `single_dot` on `succ@m` while its `unctrl_err` stays at chance and
  its `r2_unctrl` ≈ 0.
