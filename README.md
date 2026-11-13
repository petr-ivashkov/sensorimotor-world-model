# Sensorimotor World Models
### Perception for Action via Inverse Dynamics

[Petr Ivashkov](https://openreview.net/profile?id=~Petr_Ivashkov1), [Randall Balestriero](https://openreview.net/profile?id=~Randall_Balestriero1) and [Bernhard Schölkopf](https://openreview.net/profile?id=~Bernhard_Sch%C3%B6lkopf1)

<p align="center">
  <b>[ <a href="https://arxiv.org/abs/2606.20104">Paper</a> ] · [ <a href="https://petr-ivashkov.github.io/sensorimotor-world-model.github.io/">Project page</a> ] · [ <a href="https://huggingface.co/petr-ivashkov/smwm">Checkpoints</a> ]</b>
</p>

<p align="center">
  <img src="assets/schematic.png" alt="Sensorimotor world model schematic" width="90%">
</p>

We introduce a sensorimotor world model (SMWM): a latent world model trained end-to-end
from pixels with inverse dynamics regularization as the sole anti-collapse mechanism.
Empirically, SMWM learns interpretable low-dimensional representations and achieves
competitive planning performance across 2D and 3D tasks.

## Method

We train from transitions `(o_t, a_t, o_tp1)`, where `o_t` and `o_tp1` are observations
at times `t` and `t+1`. The shared `encoder` maps them to `z_t` and `z_tp1`.
The `forward_model` predicts `z_tp1` from `(z_t, a_t)`, while the `inverse_model`
recovers `a_t` from `(z_t, z_tp1)`. Their mean-squared errors form the joint loss:

```python
z_t, z_tp1 = encoder(o_t), encoder(o_tp1)
loss_fwd = mse(forward_model(z_t, a_t), z_tp1)
loss_inv = mse(inverse_model(z_t, z_tp1), a_t)
loss = loss_fwd + lambda_inv * loss_inv
```

All three components are trained jointly, with both losses backpropagating through
the encoder. `lambda_inv` weights the inverse loss; the main planning comparison uses
`lambda_inv = 10`.

## Repository structure

| Subproject | Experiments |
|---|---|
| [toy/](toy/README.md) | Dot and sprite worlds: intrinsic dimension, controllable degrees of freedom, and control-dependent reconstruction. |
| [planning/](planning/README.md) | Planning, embedding geometry, and physical-state probing on TwoRoom, Reacher, Push-T, and OGBench-Cube. |

<a id="getting-started"></a>

## Setup

A single environment at the repo root covers both subprojects:

```bash
git clone https://github.com/petr-ivashkov/sensorimotor-world-model.git
cd sensorimotor-world-model
uv sync --locked
source .venv/bin/activate
```

Python 3.12 and dependencies are pinned. Toy experiments and analysis run on CPU,
including macOS. Planning training and simulation require Linux and a CUDA GPU.

## Instructions

To try the method in dot world:

```bash
python toy/train.py --config toy/config/dot.yaml --smoke
```

Open [toy/analyze.ipynb](toy/analyze.ipynb) and select `dot_smoke`.
Omit `--smoke` for the full configuration. See the [toy guide](toy/README.md)
for the structured-dot and sprite experiments.

For planning, follow the [data preparation](planning/README.md#data), then:

```bash
cd planning
python train.py data=tworoom method=smwm seed=0
python eval.py env=tworoom run_dir=outputs/train/tworoom_smwm_seed0
```

<a id="pretrained-models-and-analysis"></a>
<a id="explore-the-pretrained-models"></a>

## Pretrained models

[Checkpoints](https://huggingface.co/petr-ivashkov/smwm) and
[analysis data](https://huggingface.co/datasets/petr-ivashkov/smwm-analysis) are available
on Hugging Face. The following downloads the planning results and saved embeddings,
plus a TwoRoom checkpoint and sample images (about 560 MB):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "petr-ivashkov/smwm-analysis", repo_type="dataset", revision="v1.0",
    local_dir="planning/outputs",
    allow_patterns=["embeddings/**", "eval/**", "samples/tworoom/*"],
)
snapshot_download(
    "petr-ivashkov/smwm", revision="v1.0",
    local_dir="planning/outputs/pretrained",
    allow_patterns=["tworoom/smwm/seed0/*", "LICENSE*"],
)
PY
```

Run `jupyter lab`, or use the `.venv` kernel in VS Code. The
[planning comparison](planning/analysis/planning_results.ipynb),
[embedding analysis](planning/analysis/embeddings.ipynb), and
[physical probes](planning/analysis/physical_probes.ipynb) run directly on these data.
See [Planning](planning/README.md#pretrained-models) to encode images with a released model.

## Citation

```bibtex
@misc{ivashkov2026sensorimotor,
  title         = {Sensorimotor World Models: Perception for Action via Inverse Dynamics},
  author        = {Ivashkov, Petr and Balestriero, Randall and Sch{\"o}lkopf, Bernhard},
  year          = {2026},
  eprint        = {2606.20104},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.20104}
}
```

## License

Released under the [MIT License](LICENSE). The planning implementation derives from
[LeWorldModel](https://github.com/lucas-maes/le-wm). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.
