"""Train the pinned DINO-WM implementation with its default objective."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault('REPO_ROOT', str(REPO_ROOT))

from baselines.dino_wm.model import build_dino_wm
from utils import get_column_normalizer, disable_external_callbacks


def get_runs_root() -> Path:
    default_root = REPO_ROOT / 'outputs' / 'train'
    return Path(os.environ.get('RUNS_ROOT', str(default_root))).resolve()


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(
        **stats, source=source, target=target
    )
    resize = spt.data.transforms.Resize(
        img_size, source=source, target=target
    )
    return spt.data.transforms.Compose(to_image, resize)


def strip_action_dims(tensor, action_range):
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def runtime_accumulation_steps(module) -> int:
    trainer = module.trainer
    return int(
        getattr(
            trainer,
            'accumulate_grad_batches_',
            trainer.accumulate_grad_batches,
        )
    )


def mean_accumulation_loss(module, objective_loss):
    accumulation = runtime_accumulation_steps(module)
    if accumulation < 1:
        raise RuntimeError(
            f'Invalid runtime gradient accumulation: {accumulation}'
        )
    return objective_loss / accumulation


def dinowm_forward(self, batch, stage, cfg):
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0)

    batch = self.model.encode(batch, target='embed')
    embedding = batch['embed'][:, : cfg.wm.history_size]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['embed'][:, cfg.wm.num_preds :].detach()

    pixels_dim = batch['pixels_embed'].size(-1)
    batch['pixels_loss'] = torch.nn.functional.mse_loss(
        pred_embedding[..., :pixels_dim],
        target_embedding[..., :pixels_dim],
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_embed'].size(-1)
        low, high = start, start + dim
        if key == 'action':
            action_range = [low, high]
        else:
            batch[f'{key}_loss'] = torch.nn.functional.mse_loss(
                pred_embedding[..., low:high],
                target_embedding[..., low:high].detach(),
            )
        start = high

    batch['actionless_embed'] = strip_action_dims(
        batch['embed'], action_range
    )
    batch['actionless_prev_embed'] = strip_action_dims(
        embedding, action_range
    )
    batch['actionless_pred_embed'] = strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_embed'] = strip_action_dims(
        target_embedding, action_range
    )
    objective_loss = torch.nn.functional.mse_loss(
        batch['actionless_pred_embed'],
        batch['actionless_target_embed'].detach(),
    )
    if objective_loss.isnan():
        raise ValueError('NaN loss encountered')
    batch['objective_loss'] = objective_loss
    batch['loss'] = (
        mean_accumulation_loss(self, objective_loss)
        if stage == 'fit'
        else objective_loss
    )

    losses = {
        f'{stage}/{key}': value.detach()
        for key, value in batch.items()
        if '_loss' in key
    }
    losses[f'{stage}/loss'] = objective_loss.detach()
    if stage == 'fit':
        self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True)
    else:
        self.log_dict(losses, on_step=False, on_epoch=True, sync_dist=True)
    return batch


def build_datasets(cfg):
    train_set = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    train_set._open()
    encoding_keys = list(cfg.wm.encoding)
    transforms = [
        get_img_preprocessor('pixels', 'pixels', int(cfg.image_size))
    ]

    with open_dict(cfg):
        cfg.extra_dims = {}
        for key in encoding_keys:
            if key not in train_set.column_names:
                raise ValueError(f"Encoding key '{key}' not in dataset")
            transforms.append(get_column_normalizer(train_set, key, key))
            input_dim = int(train_set.get_dim(key))
            if key == 'action':
                input_dim *= int(cfg.data.dataset.frameskip)
            cfg.extra_dims[key] = input_dim

    transform = spt.data.transforms.Compose(*transforms)
    train_set.transform = transform

    eval_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    train_name = eval_kwargs['name']
    if not train_name.endswith('_train'):
        raise ValueError(f'Training dataset must end in _train: {train_name}')
    eval_kwargs['name'] = train_name[: -len('_train')] + '_eval'
    val_set = swm.data.HDF5Dataset(**eval_kwargs, transform=transform)
    return train_set, val_set


def build_loaders(cfg, train_set, val_set):
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=generator
    )
    val_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val_loader = torch.utils.data.DataLoader(val_set, **val_cfg)
    return train_loader, val_loader


def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)
    run_dir = get_runs_root() / cfg.get("subdir", "")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run_dir}. Choose another subdir.")
    train_set, val_set = build_datasets(cfg)
    accumulation_steps = int(cfg.trainer.accumulate_grad_batches)
    effective_batch = int(cfg.loader.batch_size) * accumulation_steps
    updates_per_epoch = len(train_set) // effective_batch
    if updates_per_epoch < 1:
        raise ValueError("Training split is smaller than one effective batch.")
    total_updates = updates_per_epoch * int(cfg.trainer.max_epochs)
    with open_dict(cfg):
        cfg.trainer.limit_train_batches = updates_per_epoch * accumulation_steps
    scheduler = {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": max(1, int(0.01 * total_updates)),
        "max_steps": total_updates,
        "warmup_start_lr": 0.0,
        "eta_min": 0.0,
    }
    train_loader, val_loader = build_loaders(cfg, train_set, val_set)
    data_module = spt.data.DataModule(
        train=train_loader,
        val=val_loader,
    )

    model = build_dino_wm(cfg)
    with open_dict(cfg):
        cfg.backbone.config = model.backbone.config.to_dict()
    module = spt.Module(
        model=model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'optimizer': dict(cfg.optimizer),
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': accumulation_steps,
        },
    )
    module._log_hyperparams = False

    run_dir = get_runs_root() / cfg.subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')

    disable_external_callbacks()
    csv_logger = CSVLogger(
        save_dir=str(run_dir), name='lightning', version='local'
    )
    logger = csv_logger
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(save_dir=str(run_dir), **cfg.wandb.config)
        logger = [csv_logger, wandb_logger]

    trainer = pl.Trainer(
        **cfg.trainer,
        num_sanity_val_steps=1,
        default_root_dir=str(run_dir),
        logger=logger,
        enable_checkpointing=False,
    )
    ckpt_path = run_dir / 'checkpoints' / 'last.ckpt'
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=int(cfg.seed),
        ckpt_path=ckpt_path,
    )
    manager()


if __name__ == '__main__':
    import hydra

    hydra.main(version_base=None, config_path="../../config", config_name="dino_wm")(run)()
