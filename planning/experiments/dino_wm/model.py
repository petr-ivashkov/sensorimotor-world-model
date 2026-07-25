"""Construct the pinned stable-worldmodel DINO-WM implementation."""

from __future__ import annotations

import torch
from stable_worldmodel.wm.prejepa import CausalPredictor, Embedder, PreJEPA
from transformers import AutoModel


def build_dino_wm(cfg):
    backbone = AutoModel.from_pretrained(
        cfg.backbone.name,
        revision=cfg.backbone.revision,
    )
    backbone.eval()
    backbone.requires_grad_(False)

    embed_dim = int(backbone.config.hidden_size) + sum(
        int(dim) for dim in cfg.wm.encoding.values()
    )
    num_patches = (int(cfg.image_size) // int(cfg.patch_size)) ** 2
    predictor = CausalPredictor(
        num_patches=num_patches,
        num_frames=int(cfg.wm.history_size),
        dim=embed_dim,
        depth=int(cfg.predictor.depth),
        heads=int(cfg.predictor.heads),
        mlp_dim=int(cfg.predictor.mlp_dim),
        dim_head=int(cfg.predictor.dim_head),
        dropout=float(cfg.predictor.dropout),
        emb_dropout=float(cfg.predictor.emb_dropout),
    )
    extra_encoders = torch.nn.ModuleDict(
        {
            key: Embedder(
                in_chans=int(cfg.extra_dims[key]),
                emb_dim=int(target_dim),
            )
            for key, target_dim in cfg.wm.encoding.items()
        }
    )
    return PreJEPA(
        encoder=backbone,
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=int(cfg.wm.history_size),
        num_pred=int(cfg.wm.num_preds),
        interpolate_pos_encoding=bool(
            cfg.backbone.interpolate_pos_encoding
        ),
    )
