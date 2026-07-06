# Multistep inverse dynamics + oracle-subgoal goal reaching

Extends the joint objective `L = L_fwd + λ·L_inv` to a horizon-k inverse term
(`train_multistep.py`):

```
L = ||g_ϕ(z_t, a_t) − z_{t+1}||² + λ·||h_ψ(z_t, z_{t+k}) − ā_t||²,   ā_t = (1/k)·Σᵢ a_{t+i}
```

The inverse head decodes the segment *endpoints* into the **mean action** over
the horizon — not the action sequence, which the endpoints don't identify
(multimodal target). The forward loss stays single-step. `horizon_k: 1`
reproduces `train.py` exactly (same data RNG sequence, same losses, same update
sequence — see the tests below).

Trained models are then evaluated on **oracle-subgoal goal reaching**
(`eval_goal_reaching.py`): a ground-truth episode `s_0 … s_T` is rolled with
the data-generating dynamics, every `spacing`-th state is kept as a subgoal,
and the agent decodes subgoals pairwise, `â = h_ψ(f_θ(o_cur), f_θ(o_subgoal))`,
under receding-horizon replanning (re-encode + re-decode after every executed
step; advance on arrival or budget). Subgoals are deliberately oracle states:
this isolates *decoding* from subgoal generation, so failures are attributable.
Success is scored on the controllable DOFs only; RANDOM dots keep moving during
execution and their residual error is reported separately.

## Runs

5 environments × k ∈ {1, 2, 4, 8} — `config_<env>_k<k>.yaml`:

| env | world | action_dim | question it answers |
|---|---|---|---|
| `single_dot` | 1 independent dot | 2 | base case |
| `2_independent` | 2 independent dots | 4 | multiple controllable DOFs |
| `1_coupled_pair` | 1 coupled pair | 2 | shared-action structure |
| `1_dot_1_random` | 1 dot + 1 random | 2 | is the distractor ignored? |
| `sprite_xyt` | full-control sprite (x, y, θ) | 3 | continuous pose, θ wrap |

Note: `sprite_xyt` keeps the matched-rim-speed `max_delta_theta` (~61°/step,
the `sprite_base` default) instead of `sprite_decoder/config_xyt`'s ±π: with ±π
steps the k-step mean-θ target wraps (unidentifiable) already at k=2. With ~61°
the wrap sets in gradually across the k sweep, which is the effect under study.

## Run (cluster)

```bash
# all 20 training runs, then all 5 eval configs, then the summary table
./run.sh                                   # or: ./run.sh config_single_dot_k4.yaml
./run_eval.sh                              # or: ./run_eval.sh config_eval_single_dot.yaml
python aggregate_results.py                # prints table, writes results/summary.csv
```

Each training run writes the standard layout
(`results/<run>/{model.pt, config.yaml, train_history.pt, embeddings.pt}`), so
`train_decoder.py` and the existing notebooks work on these runs unchanged;
the eval adds `goal_reaching.pt` (per-spacing summaries + per-episode records
and traces).

**Resources.** Dot-world runs: 100 epochs; sprite: 200. Datasets are
materialized in RAM like `train.py`, but segments carry 3 frames instead of 2:
≈ **37 GB** resident per run at the default 250 k train samples, transiently
~2× while stacking (`train.py`: ≈ 25 GB / ~2×). If a node can't hold that,
lower `dataset.train_samples` — runs stay internally comparable as long as all
of them use the same value. Eval and aggregation are cheap and CPU-friendly:
minutes per dot-world run; sprite runs are render-bound and can take
~10–20 min each at the default 100 episodes.

## Verify

```bash
cd ../..                                   # toy/
python tests/test_multistep_inverse.py     # ~2-3 min on CPU (also runs under pytest)
```

The tests pin down, in order: (1) segment datasets at k=1 are **bitwise
identical** to the transition datasets in both worlds; (2) segment actions
match their transitions and k-step segments stay in-canvas; (3)
`train_multistep.compute_losses` at k=1 equals `train.compute_losses` exactly;
(4) a short `train_multistep.py` k=1 run produces **bitwise-identical weights**
to `train.py`; (5) replaying a segment's actions through the closed-loop
steppers reproduces its controllable state; (6) the goal-reaching eval runs
end-to-end on a smoke checkpoint and a trained smoke model beats a random-init
one.

## Reading the results

- **Does multistep IDR still recover the controllable DOFs?**
  `aggregate_results.py`: `eff_rank` should sit near `true_dim` and `r2_ctrl`
  near 1 across k; `r2_unctrl` stays low. Latent geometry / PCA plots from
  `experiments/structured_dots/analyze.ipynb` apply as-is (point `RUNS` at
  these run names).
- **Does oracle decoding still reach goals as subgoal spacing grows?**
  `succ@<spacing>` columns per run; spacing ≈ horizon_k is the matched
  condition, spacing ≫ k probes out-of-distribution endpoint gaps.
- **Does planning ignore the uncontrollable dot?** `1_dot_1_random` should
  track `single_dot` on `succ@m` / `pos_err@m` while its `unctrl_err`
  (distance of the random dot to its goal position) stays at chance level.
