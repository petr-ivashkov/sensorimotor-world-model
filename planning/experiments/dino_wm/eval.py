"""Run shared planning evaluation with a strictly loaded DINO-WM."""

from __future__ import annotations

import sys
from pathlib import Path

import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import eval as shared_eval
from model import build_dino_wm


EXPECTED_PROTOCOL = 'matched_batch256_mean_accumulation_v2'


def load_dino_wm_from_run(run_dir: Path, device: str = 'cuda'):
    run_dir = Path(run_dir)
    train_cfg = OmegaConf.load(run_dir / 'config.yaml')
    protocol = str(
        train_cfg.get('optimization_matching', {}).get(
            'protocol_version',
            '',
        )
    )
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f'Checkpoint uses DINO-WM protocol {protocol!r}; expected '
            f'{EXPECTED_PROTOCOL!r}'
        )
    model = build_dino_wm(train_cfg)
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
    model.interpolate_pos_encoding = True
    return model, train_cfg


def dino_img_transform(img_size: int):
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
    shared_eval.load_jepa_from_run = load_dino_wm_from_run
    shared_eval.img_transform = dino_img_transform
    shared_eval.main()
