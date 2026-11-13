# Baseline implementations

We compare inverse dynamics with alternative anti-collapse objectives and frozen visual
pretraining. All methods use the same held-out planning tasks, goal offset, and step budget.
Commands below run from `planning/`.

## SMWM, SIGReg, and Forward-only

These methods share the encoder, predictor, and optimization settings.
SMWM uses inverse-loss weight `λ=10`. SIGReg disables the inverse loss and applies
Gaussian-matching regularization with weight 0.09. Forward-only uses the prediction
loss alone (`λ=0`), isolating the contribution of regularization.

```bash
python train.py data=tworoom method=smwm seed=0
python train.py data=tworoom method=sigreg seed=0
python train.py data=tworoom method=forward_only seed=0
```

## PLDM

We use the `stable-worldmodel` PLDM implementation with a ViT-Tiny encoder trained
from scratch. The objective combines prediction, standard deviation, temporal standard
deviation, covariance, and temporal alignment losses with weights 1, 18, 0.7, 12, and 0.2.
The inverse-dynamics coefficient is zero. See [pldm.yaml](../config/pldm.yaml).

```bash
python baselines/pldm/train.py data=tworoom seed=0
python eval.py env=tworoom run_dir=outputs/train/tworoom_pldm_seed0
```

## DINO-WM

We use `stable_worldmodel.wm.prejepa` with a frozen, pretrained DINOv2-Small patch
encoder. The causal predictor and action encoder are trained from the environment
transitions; the prediction loss excludes action channels. Inputs are pixels and
actions, without proprioceptive or simulator-state inputs. See [dino_wm.yaml](../config/dino_wm.yaml).

```bash
python baselines/dino_wm/train.py data=tworoom seed=0
python eval.py env=tworoom run_dir=outputs/train/tworoom_dino_wm_seed0
```

For PLDM and DINO-WM, predictor history is reduced from 3 to 1 to match the comparison.
Both use the shared data splits, ten epochs, and effective batch size 256.
DINO-WM uses eight mean-reduced micro-batches of 32 and an optimizer-update-based schedule.

## Random

Random samples actions uniformly from the action space. The five reported runs are
policy repeats, evaluated on the same tasks and step budget as the learned methods.

```bash
python eval.py env=tworoom policy=random repeat=0
```

The [planning notebook](../analysis/planning_results.ipynb) summarizes all six methods.
PLDM and DINO-WM are adapted through
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) from
[PLDM](https://github.com/vladisai/PLDM) and [DINO-WM](https://github.com/gaoyuezhou/dino_wm).
See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for licenses and attribution.
