"""Train the official PLDM architecture and default objective on matched data."""

from __future__ import annotations

import os
import shutil
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import lightning.fabric.utilities.registry as lightning_registry
import lightning.pytorch.trainer.connectors.callback_connector as callback_connector
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault('REPO_ROOT', str(REPO_ROOT))

from utils import get_column_normalizer, save_embeddings
from vendor.loss import PLDMLoss, TemporalStraighteningLoss


def get_runs_root() -> Path:
    default_root = EXPERIMENT_DIR / 'results' / 'train'
    return Path(os.environ.get('RUNS_ROOT', str(default_root))).resolve()


def configure_external_callbacks(enabled: bool) -> None:
    if enabled:
        return

    def no_external_callbacks(_group: str):
        return []

    lightning_registry._load_external_callbacks = no_external_callbacks
    callback_connector._load_external_callbacks = no_external_callbacks


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = spt.data.transforms.Resize(
        img_size, source=source, target=target
    )
    return spt.data.transforms.Compose(to_image, resize)


def pldm_forward(self, batch, stage, cfg):
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)

    emb = output['emb']
    act_emb = output['act_emb']
    inpt_emb = emb[:, : cfg.wm.history_size]
    inpt_act = act_emb[:, : cfg.wm.history_size]
    tgt_emb = emb[:, cfg.wm.num_preds :]
    pred_emb = self.model.predict(inpt_emb, inpt_act)

    output['idm_emb'] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output['act_label'] = batch['action'][:, :-1].detach()
    output['act_pred'] = self.idm(output['idm_emb'])
    output['pred_loss'] = (pred_emb - tgt_emb).square().mean()
    output['temp_straight_loss'] = self.path_straight(emb)
    output.update(self.pldm(emb, output['act_pred'], output['act_label']))

    output['loss'] = output['pred_loss']
    for key, loss_cfg in cfg.loss.items():
        loss_key = f'{key}_loss'
        if not loss_cfg.enabled or loss_key not in output:
            continue
        output['loss'] = output['loss'] + loss_cfg.weight * output[loss_key]

    losses = {
        f'{stage}/{key}': value.detach()
        for key, value in output.items()
        if 'loss' in key
    }
    if stage == 'fit':
        self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True)
    else:
        self.log_dict(losses, on_step=False, on_epoch=True, sync_dist=True)
        output['emb'] = emb.detach()
    return output


def build_datasets(cfg):
    train_set = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    train_set._open()

    transforms = [get_img_preprocessor('pixels', 'pixels', cfg.img_size)]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if column == 'pixels':
                continue
            transforms.append(
                get_column_normalizer(train_set, column, column)
            )
            setattr(cfg.wm, f'{column}_dim', train_set.get_dim(column))

        action_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
        cfg.model.action_encoder.input_dim = action_dim
        cfg.idm.input_dim = 2 * cfg.wm.embed_dim
        cfg.idm.output_dim = action_dim

    transform = spt.data.transforms.Compose(*transforms)
    train_set.transform = transform

    eval_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    train_name = eval_kwargs['name']
    if not train_name.endswith('_train'):
        raise ValueError(f'Training dataset must end in _train: {train_name}')
    eval_kwargs['name'] = train_name[: -len('_train')] + '_eval'
    val_set = swm.data.HDF5Dataset(**eval_kwargs, transform=transform)
    return train_set, val_set


@hydra.main(version_base=None, config_path=None, config_name=None)
def run(cfg):
    train_set, val_set = build_datasets(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=generator
    )
    val_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val_loader = torch.utils.data.DataLoader(val_set, **val_cfg)
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    model = hydra.utils.instantiate(cfg.model)
    idm = hydra.utils.instantiate(cfg.idm)
    with open_dict(cfg):
        cfg.pldm_experiment.optimizer_updates_per_epoch = len(train_loader)
    optimizers = {}
    for model_name in ('model', 'idm'):
        optimizers[f'{model_name}_opt'] = {
            'modules': model_name,
            'optimizer': dict(cfg.optimizer),
            'scheduler': 'LinearWarmupCosineAnnealingLR',
            'interval': 'epoch',
        }

    module = spt.Module(
        model=model,
        idm=idm,
        pldm=PLDMLoss(),
        path_straight=TemporalStraighteningLoss(),
        forward=partial(pldm_forward, cfg=cfg),
        optim=optimizers,
    )
    module._log_hyperparams = False

    run_dir = get_runs_root() / cfg.subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')

    configure_external_callbacks(
        cfg.get('artifacts', {}).get('use_external_callbacks', False)
    )
    lightning_dir = run_dir / 'lightning' / 'local'
    if lightning_dir.exists():
        shutil.rmtree(lightning_dir)

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
        seed=cfg.seed,
        ckpt_path=ckpt_path,
    )
    manager()

    barrier = getattr(manager._trainer.strategy, 'barrier', None)
    if barrier is not None:
        barrier('save_embeddings_start')
    if manager._trainer.is_global_zero:
        save_embeddings(
            manager.instantiated_module.model,
            val_set,
            run_dir / 'final_embeddings.pt',
            batch_size=int(cfg.loader.batch_size),
            max_items=cfg.artifacts.embedding_subset_size,
        )
    if barrier is not None:
        barrier('save_embeddings_end')


if __name__ == '__main__':
    run()
