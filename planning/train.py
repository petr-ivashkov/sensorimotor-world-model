"""
Training script for latent world model with forward + inverse objectives:
    L = L_fwd + lambda_inv * L_inv
An optional SIGReg term can be enabled explicitly for comparisons.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, InverseModel, MLP, SIGReg
from utils import ResizeCompat, get_column_normalizer, disable_external_callbacks

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))


def get_runs_root() -> Path:
    default_root = REPO_ROOT / "outputs" / "train"
    return Path(os.environ.get("RUNS_ROOT", str(default_root))).expanduser().resolve()


def forward_step(self, batch, stage, cfg):
    lambd_sigreg = cfg.loss.sigreg.weight
    lambd_inv = cfg.loss.inverse.weight
    history_size = int(cfg.wm.get("history_size", 1))
    required_steps = history_size + 1

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    if emb.size(1) < required_steps:
        raise ValueError(
            f"Batch sequence length {emb.size(1)} is too short for "
            f"wm.history_size={history_size}; need at least {required_steps} "
            "steps for one-step supervision."
        )

    pred_emb = self.model.predict(emb[:, :history_size], act_emb[:, :history_size])
    tgt_emb = emb[:, 1:required_steps]

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    output["loss"] = output["pred_loss"]
    if lambd_inv:
        z_t = emb[:, :-1]
        z_tp1 = emb[:, 1:]
        actions = batch["action"][:, :-1]
        pred_actions = self.model.predict_action(z_t, z_tp1)
        output["inv_loss"] = (pred_actions - actions).pow(2).mean()
        output["loss"] = output["loss"] + lambd_inv * output["inv_loss"]

    if lambd_sigreg:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["loss"] + lambd_sigreg * output["sigreg_loss"]

    if stage != "fit":
        output["emb"] = emb.detach()

    logs = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if stage == "fit":
        self.log_dict(logs, on_step=True, on_epoch=True, sync_dist=True)
    else:
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="config", config_name="train")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)
    run_dir = get_runs_root() / cfg.get("subdir", "")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run_dir}. Choose another subdir.")
    history_size = int(cfg.wm.get("history_size", 1))
    required_steps = history_size + 1
    with open_dict(cfg):
        cfg.wm.history_size = history_size
        cfg.wm.num_preds = int(cfg.wm.get("num_preds", 1))
        current_steps = int(cfg.data.dataset.get("num_steps", required_steps))
        if current_steps < required_steps:
            cfg.data.dataset.num_steps = required_steps

    train_set = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    # Read only the physical pixel column; merged cached keys may not exist in HDF5.
    train_set._open()
    pixel_shape = train_set.h5_file["pixels"][0].shape
    pixel_hw = tuple(pixel_shape[:2]) if len(pixel_shape) >= 2 else ()
    img_size = None if pixel_hw == (cfg.img_size, cfg.img_size) else cfg.img_size
    transforms = []
    if img_size is not None:
        transforms.append(ResizeCompat(img_size, source="pixels", target="pixels"))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(train_set, col, col))
            setattr(cfg.wm, f"{col}_dim", train_set.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    train_set.transform = transform

    eval_kwargs = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if not eval_kwargs["name"].endswith("_train"):
        raise ValueError(
            f"cfg.data.dataset.name={eval_kwargs['name']!r} must end with "
            "'_train' so the held-out '_eval' split name can be derived"
        )
    eval_kwargs["name"] = eval_kwargs["name"][: -len("_train")] + "_eval"
    val_set = swm.data.HDF5Dataset(**eval_kwargs, transform=transform)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    world_model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        ),
        action_encoder=Embedder(input_dim=effective_act_dim, emb_dim=embed_dim),
        projector=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        inverse_model=InverseModel(
            embed_dim=embed_dim,
            action_dim=effective_act_dim,
            hidden_dim=cfg.inverse.get("hidden_dim", 256),
        ),
    )

    module_kwargs = {
        "model": world_model,
        "forward": partial(forward_step, cfg=cfg),
        "optim": {
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "scheduler": "LinearWarmupCosineAnnealingLR",
                "interval": "epoch",
            }
        },
    }
    if cfg.loss.sigreg.weight:
        module_kwargs["sigreg"] = SIGReg(**cfg.loss.sigreg.kwargs)

    module = spt.Module(**module_kwargs)
    module._log_hyperparams = False
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    disable_external_callbacks()

    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning", version="local")
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

    ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=ckpt_path,
        compile=False,
    )
    manager()


if __name__ == "__main__":
    run()
