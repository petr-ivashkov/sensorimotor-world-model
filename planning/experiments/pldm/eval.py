"""Run the shared planning evaluation with a strictly loaded PLDM checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import eval as shared_eval
from planning_model import PlanningPLDM


def build_pldm(cfg):
    return PlanningPLDM(
        encoder=hydra.utils.instantiate(cfg.model.encoder),
        predictor=hydra.utils.instantiate(cfg.model.predictor),
        action_encoder=hydra.utils.instantiate(cfg.model.action_encoder),
        projector=hydra.utils.instantiate(cfg.model.projector),
        pred_proj=hydra.utils.instantiate(cfg.model.pred_proj),
    )


def load_pldm_from_run(run_dir: Path, device: str = 'cuda'):
    run_dir = Path(run_dir)
    train_cfg = OmegaConf.load(run_dir / 'config.yaml')
    model = build_pldm(train_cfg)
    checkpoint = torch.load(
        run_dir / 'checkpoints' / 'last.ckpt',
        map_location='cpu',
        weights_only=False,
    )
    state = {
        key[len('model.') :]: value
        for key, value in checkpoint['state_dict'].items()
        if key.startswith('model.')
    }
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, train_cfg


def pldm_img_transform(img_size: int):
    stats = spt.data.dataset_stats.ImageNet
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=stats['mean'], std=stats['std']),
            v2.Resize(size=img_size),
        ]
    )


if __name__ == '__main__':
    shared_eval.load_jepa_from_run = load_pldm_from_run
    shared_eval.img_transform = pldm_img_transform
    shared_eval.main()
