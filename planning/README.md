# Planning experiments

Reproduces the paper's planning results on four environments — **TwoRoom, Reacher,
Push-T, and OGBench-Cube**.

## Method

Three components are trained jointly:

- **Encoder** `f_theta`: observation -> latent `z` (ViT-Tiny, CLS-token pooling + MLP projector)
- **Forward model** `g_phi`: `(z_t, a_t) -> z_hat_{t+1}` (autoregressive transformer, AdaLN-zero action conditioning)
- **Inverse model** `h_psi`: `(z_t, z_{t+1}) -> a_hat_t` (MLP)

```text
L = L_fwd + lambda * L_inv
```

`L_fwd` is the MSE between the predicted next-state embedding and the encoder's own embedding of
the actual next observation. `L_inv` is the MSE between
predicted and executed actions. An optional SIGReg term is available for the baseline
comparison but is off by default. Per-environment `lambda` values are in the paper (Table 1).

## Setup

The environment is defined once at the repo root (see the top-level [`README.md`](../README.md)).
The planning stack requires a single CUDA GPU and installs on Linux (~4 h per training run):

```bash
uv sync                 # from the repo root; creates ../.venv
source ../.venv/bin/activate
```

`run.sh` in each experiment stage activates this environment automatically.

## Data

All datasets live under `data/external/` (gitignored). They come from the LeWM HuggingFace
collection: <https://huggingface.co/collections/quentinll/lewm>.

Push-T example:

```bash
mkdir -p data/external
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import zstandard as zstd

out = Path("data/external")
zst = Path(hf_hub_download(repo_id="quentinll/lewm-pusht", repo_type="dataset",
                           filename="pusht_expert_train.h5.zst", local_dir=out))
dst = out / "pusht_expert_train.h5"
with zst.open("rb") as s, dst.open("wb") as d:
    zstd.ZstdDecompressor().copy_stream(s, d)
print("wrote", dst)
PY
```

Download the expert train datasets for all four environments into `data/external/` with these
names (the TwoRoom / Reacher / OGBench-Cube datasets are in the same collection — confirm the
exact `repo_id`s for your release):

```text
tworoom_train.h5   reacher_train.h5   pusht_expert_train.h5   cube_single_expert_train.h5
```

Then build the episode splits and the randomized transition subsets used by the analyses:

```bash
python scripts/make_episode_splits.py                    # train/eval splits, e.g. tworoom_eval.h5
python scripts/create_randomized_transition_datasets.py  # 25k/5k subsets for embedding_analysis
```

## Repository layout

```text
jepa.py  module.py  train.py  eval.py  utils.py  plot_style.py   # model + training/eval/analysis
config/
    train/{base.yaml, data/<env>.yaml}      # training configs (base + per-env overrides)
    eval/{base.yaml,  env/<env>.yaml}       # planning-eval configs
scripts/                                    # dataset split / subset helpers
experiments/                                # one folder per pipeline stage (see its README)
    train/                                  # 4 envs x 3 methods x 5 seeds   -> checkpoints
    planning_eval/                          # Fig. 5
    horizon_sweep/                          # Fig. 9
    lambda_sweep/                           # Fig. 7 + Table 1
    embedding_analysis/                     # Fig. 6, Fig. 10, Table 2
data/                                       # datasets (gitignored)
```

Each stage has a `README.md` with its exact commands, a portable `run.sh` launcher, and (where
it owns a figure) a notebook that displays results inline.

## Reproduce

Run the stages in dependency order. `generate_configs.py` in `experiments/train` must run once
first — it writes the `manifest.tsv` that `planning_eval` and `horizon_sweep` read. Every figure
renders in its notebook.

```text
train ───┬──> planning_eval
         ├──> horizon_sweep
         └──> embedding_analysis

lambda_sweep/train_<env> ──> lambda_sweep/eval_<env>
```

Launch each job through the stage's `run.sh` (see the stage's README for the exact arguments),
then aggregate the metrics and **Run All** in the notebook:

```bash
cd experiments/train && python generate_configs.py
./run.sh tworoom_inverse_lambda_0p1_seed0      # repeat per run name in manifest.tsv
```

| Notebook | Paper output |
|---|---|
| `experiments/planning_eval/plot_results.ipynb` | Fig. 5 |
| `experiments/horizon_sweep/plot_results.ipynb` | Fig. 9 |
| `experiments/lambda_sweep/plot_results.ipynb` | Fig. 7 + Table 1 |
| `experiments/embedding_analysis/analyze_learned_representations.ipynb` | Fig. 6, Fig. 10 |
| `experiments/embedding_analysis/physical_quantity_probe_table.ipynb` | Table 2 |

## Credit

This subproject derives from [LeWorldModel](https://github.com/lucas-maes/le-wm)
(LeWM) by Lucas Maes, MIT-licensed; the files `jepa.py`, `module.py`, `train.py`, `eval.py`,
`utils.py` and the `config/` Hydra layout are adapted and extended from it.
