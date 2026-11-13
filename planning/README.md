# Planning experiments

SMWM and its baselines on **TwoRoom, Reacher, Push-T, and OGBench-Cube**.
The experiments evaluate goal-conditioned planning and examine the physical structure
retained by the learned representations.

Use the root [setup](../README.md#setup). Training and simulation require Linux and
a CUDA GPU; analysis runs on CPU. Commands below start in `planning/`.

## Method

Three components are trained jointly:

- **Encoder** `f`: maps an image to a latent state (ViT-Tiny, CLS token and MLP projector).
- **Forward model** `g`: predicts the next latent from the current latent and action.
- **Inverse model** `h`: recovers the action from consecutive latents.

The objective is `L = L_fwd + λ·L_inv`. The main comparison fixes `λ=10` and predictor
history 1 in every environment. The encoder and forward model follow LeWorldModel;
the inverse model is a two-layer MLP.

## Planning comparison

Given a current image and a goal image, we optimize action sequences through latent
rollouts of the frozen world model using CEM. We compare SMWM with SIGReg, PLDM,
DINO-WM, Forward-only, and Random; see [baseline implementations](baselines/README.md).

```bash
python train.py data=tworoom method=smwm seed=0
python eval.py env=tworoom run_dir=outputs/train/tworoom_smwm_seed0
```

Choose `tworoom`, `reacher`, `pusht`, or `ogbcube` with `data=` and `env=`.
SMWM, SIGReg, and Forward-only share `train.py`; select them with `method=`.
Random requires no checkpoint: `python eval.py env=tworoom policy=random repeat=0`.

The paper comparison uses training seeds 0–4, ten epochs, and effective batch size 256.
Each model is evaluated on 100 tasks with goal offset 25 and a budget of 50 environment
steps. The figure reports mean ± sample STD across training seeds; Random uses five
policy repeats. The [planning notebook](analysis/planning_results.ipynb) produces the
figure and numerical table.

## Representation analysis

The [embedding notebook](analysis/embeddings.ipynb) shows PCA spectra, physical-state
distributions, and PC projections of the embeddings. Corresponding physical states and
embeddings share the same colors. This reveals how the latent geometry relates to
spatial coordinates and periodic joint angles.

The [physical-probe notebook](analysis/physical_probes.ipynb) measures how accurately
positions, angles, and gripper state can be decoded from frozen embeddings. Linear and
two-layer MLP probes are trained on one split and evaluated on another, with held-out
`R²` as the metric.

<a id="using-a-pretrained-model"></a>

## Pretrained models

The root [download instructions](../README.md#pretrained-models) provide the saved
planning results and seed-0 embeddings for SMWM, SIGReg, and Forward-only.
[Checkpoints](https://huggingface.co/petr-ivashkov/smwm) are available for all five
learned methods, environments, and training seeds.

To encode the sample images with the downloaded TwoRoom model:

```bash
python scripts/export_embeddings.py \
  --run-dir outputs/pretrained/tworoom/smwm/seed0 \
  --train-data outputs/samples/tworoom/train.h5 \
  --eval-data outputs/samples/tworoom/eval.h5 \
  --output-dir outputs/demo_embeddings --batch-size 32 --device cpu
```

Set `EXPORT_ROOT = PLANNING / "outputs/demo_embeddings"` in an analysis notebook to
inspect this export. To select another checkpoint, change `allow_patterns` in the
download command. The evaluation script also accepts downloaded checkpoint directories.

<a id="training-data"></a>

## Data

The offline expert datasets come from [LeWM](https://huggingface.co/collections/quentinll/lewm).
Download and decompress them into `data/external/`:

| Environment | Download | File name |
|---|---|---|
| TwoRoom | [tworoom.tar.zst](https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/6903a2de048b13819d812da0b4dd661290bc01e4/tworoom.tar.zst) | `tworoom.h5` |
| Reacher | [reacher.tar.zst](https://huggingface.co/datasets/quentinll/lewm-reacher/resolve/e70a080d0d04c6072123c9ebd343acf7fff28dbf/reacher.tar.zst) | `reacher.h5` |
| Push-T | [pusht_expert_train.h5.zst](https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst) | `pusht_expert.h5` |
| OGBench-Cube | [cube_single_expert.tar.zst](https://huggingface.co/datasets/quentinll/lewm-cube/resolve/02a19a67a0dc8c9d6215f89c19e0a597691e152a/cube_single_expert.tar.zst) | `cube_single_expert.h5` |

Use `tar --zstd -xf archive.tar.zst` for the archives and
`zstd -d pusht_expert_train.h5.zst -o pusht_expert.h5` for Push-T.
Prepare the episode-level training/evaluation split:

```bash
python scripts/make_episode_splits.py
```

The split is 90/10 with seed 3072. Existing splits are preserved.
Set `EXTERNAL_DATA_ROOT` to use another directory.

For representation analysis with a newly trained model:

```bash
python scripts/create_randomized_transition_datasets.py --train-n 25000 --eval-n 5000
python scripts/export_embeddings.py --run-dir outputs/train/tworoom_smwm_seed0
```

The full analysis images are also available under `images/<environment>/` in the
[released dataset](https://huggingface.co/datasets/petr-ivashkov/smwm-analysis).
The [configurations](config/) expose the loss weights, predictor history, and planning settings.

## Credit

Adapted from [LeWorldModel](https://github.com/lucas-maes/le-wm), using
`stable-worldmodel` and `stable-pretraining`. See the
[third-party notices](../THIRD_PARTY_NOTICES.md).
