# Online adaptation on a U-shaped block

Continue training a pretrained Push-T SMWM while collecting experience with a
new U-shaped movable block. The current task is to establish a working online
setting and inspect representation and prediction changes. Planning evaluation
is disabled in the small U run.

The U has a cavity the pusher can enter, a 120×120 exterior, and the same mass as
the original T. It preserves the existing two-dimensional actions and
seven-dimensional simulator state. `shape_env.py` registers `swm/PushT-U-v0`
without changing the installed stable-worldmodel package.

The center-of-gravity experiment is retired. Its successful smoke verified the
basic collect/update/evaluate/checkpoint loop, but it did not validate the new
U environment. Full U reset, stepping, and rendering remain to be checked on the
cluster. No U job has run yet.

## How the loop works

Each round collects episodes, adds them to replay, takes a configured number of
updates, and saves a checkpoint plus diagnostics on fixed held-out clips. The
model and AdamW state continue across rounds. Original datasets and checkpoints
are read in place; newly collected experience and results belong to this
experiment's output directory.

The default collector is state-informed random pushing. At each raw step it
samples a displacement around the pusher, constrains the desired position toward
a square around the block, and converts it into a bounded relative control.
It uses simulator positions rather than the world model and does not pursue a
goal. A new action is drawn every raw step.

Training preserves the pretrained temporal convention: two images five raw
steps apart, with the intervening five normalized 2D actions concatenated into
a 10D action token. Source-training action statistics remain fixed. With
`z = projector(encoder(image))`, the loss is:

```text
L_forward = mean((predicted_z_next - z_next)^2)
L_inverse = mean((predicted_actions - recorded_actions)^2)
L = L_forward + source_inverse_weight * L_inverse
```

The selected source checkpoint uses inverse weight 10 and no SIGReg. Every
model component is trainable. Neither encoded branch is detached: both losses
send gradients into the shared encoder/projector through both observations.
Rewards, goals, physical-state probes, and planning success do not train SMWM.
BatchNorm running statistics also update during training forward passes.

Diagnostics compare fixed image embeddings and one-/multi-step predictions
before and after updates. Physical-state probes fit separate episodes and stay
outside model training. CKA/drift measure change, while prediction errors help
assess whether that change is useful.

## Git and cluster workflow

Use the `research/online-adaptation` branch in a proper Git checkout/worktree.
The intended cluster worktree is
`~/projects/sensorimotor-world-model-online-adaptation`. Commit and push changes
locally, then update that branch on the cluster before running an experiment.
The original checkout supplies its existing venv, pretrained model, and data.
There is no code-bundle deployment step.

From the cluster research worktree:

```bash
export ORIGINAL_REPO=/lustre/home/pivashkov/projects/sensorimotor-world-model
export SMWM_PYTHON="$ORIGINAL_REPO/.venv/bin/python"
export EXTERNAL_DATA_ROOT="$ORIGINAL_REPO/planning/data/external"
cd ~/projects/sensorimotor-world-model-online-adaptation/planning/experiments/online_adaptation

# Read prerequisites on the login node; no training or simulation.
bash run.sh --config smoke.yaml \
  --source-run "$ORIGINAL_REPO/planning/experiments/train/results/pusht_inverse_lambda_10_seed0" \
  --source-data "$EXTERNAL_DATA_ROOT/pusht_expert_train.h5" \
  --source-eval "$EXTERNAL_DATA_ROOT/pusht_expert_eval.h5" \
  --device cpu --preflight

# Run from a GPU allocation.
bash run.sh --config smoke.yaml \
  --source-run "$ORIGINAL_REPO/planning/experiments/train/results/pusht_inverse_lambda_10_seed0" \
  --source-data "$EXTERNAL_DATA_ROOT/pusht_expert_train.h5" \
  --source-eval "$EXTERNAL_DATA_ROOT/pusht_expert_eval.h5" \
  --output-root "$PWD/results/u_smoke"
```

For HTCondor, create the results parent and submit the small configuration:

```bash
mkdir -p results
condor_submit_bid 100 smoke.sub \
  source_run="$ORIGINAL_REPO/planning/experiments/train/results/pusht_inverse_lambda_10_seed0" \
  source_data="$EXTERNAL_DATA_ROOT/pusht_expert_train.h5" \
  source_eval="$EXTERNAL_DATA_ROOT/pusht_expert_eval.h5" \
  output_root="$PWD/results/u_smoke"
```

`smoke.yaml` is the small U configuration; `pilot.yaml` defines the longer
multi-round configuration. The runner rejects nonempty output directories.
Repeat the same command with `--resume` to recover an interrupted run; changed
code/configuration requires a new output directory. No dependencies are installed
or changed by the launcher.

## Results and checks

The output contains immutable experience shards, per-round JSON metrics,
fixed anchor images/embeddings, model snapshots, and `online_state.pt` with
optimizer/RNG state. `experience_transitions` counts newly collected training
transitions; held-out acquisition is additional.

Plot exported metrics locally without moving datasets or checkpoints:

```bash
python -m planning.experiments.online_adaptation.plot_results /path/to/results/u_smoke
```

The output is `adaptation_curves.pdf`; disabled planning panels have no recorded
measurements. Run local checks with an existing environment:

```bash
python -m unittest discover -s planning/experiments/online_adaptation/tests -v
```

All 68 local tests have passed. They cover training/gradient behavior, replay,
checkpoint recovery, HDF5 reader handoff, diagnostics, and U geometry. A Pymunk
physics test verifies that the pusher enters the cavity and moves the block by
pushing its inner base. The local Pymunk version cannot run the upstream reset
callback unchanged; full U simulator validation is still pending on the cluster.
