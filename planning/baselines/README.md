# Baselines

SMWM, SIGReg, and Forward-only share `train.py`; select them with `method=`.
PLDM and DINO-WM use their adapters below, followed by the shared evaluation:

```bash
python baselines/pldm/train.py data=tworoom seed=0
python baselines/dino_wm/train.py data=tworoom seed=0
python eval.py env=tworoom run_dir=outputs/train/tworoom_pldm_seed0
python eval.py env=tworoom run_dir=outputs/train/tworoom_dino_wm_seed0
```

All methods use the same splits, history 1, 10 epochs, effective batch size 256,
and CEM tasks/budget. Both adapters use ImageNet normalization and pixels/actions
without privileged-state inputs. Full settings are in
[`pldm.yaml`](../config/pldm.yaml) and [`dino_wm.yaml`](../config/dino_wm.yaml).

- **PLDM:** reference architecture and prediction + 18 std + 0.7 std_t + 12 cov
  + 0.2 temporal-alignment loss, with history reduced from 3 to 1. Vendored from
  [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel/tree/0baacc8118c5f262aeff942da75cf191e71d3c5f/stable_worldmodel/wm/pldm), adapting [PLDM](https://github.com/vladisai/PLDM).
- **DINO-WM:** frozen DINOv2-Small with trainable predictor/action encoder;
  non-action feature MSE, history reduced from 3 to 1. Eight micro-batches of 32
  use mean-reduced gradients and an optimizer-update-based schedule. Uses
  `stable_worldmodel.wm.prejepa`, adapting [DINO-WM](https://github.com/gaoyuezhou/dino_wm).

See [third-party notices](../../THIRD_PARTY_NOTICES.md) for licenses.
