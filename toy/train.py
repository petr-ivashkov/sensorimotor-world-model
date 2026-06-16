from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from datasets.structured_dot_world import (
    DotGroup,
    MotionType,
    StructuredDotWorldConfig,
    StructuredDotWorldDataset,
    _auto_colors,
)
from datasets.sprite_world import (
    ControlConfig,
    Shape,
    SpriteWorldConfig,
    SpriteWorldDataset,
)
from models import CNNEncoder, ForwardModel, InverseModel


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"

MOTION_MAP = {
    "independent": MotionType.INDEPENDENT,
    "static": MotionType.STATIC,
    "random": MotionType.RANDOM,
    "coupled": MotionType.COUPLED,
}


# ─────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────

def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for ref in cfg.pop("defaults", []):
        merged = deep_merge(merged, yaml.safe_load((CONFIG_DIR / f"{ref}.yaml").read_text()) or {})
    return deep_merge(merged, cfg)


def build_structured_config(ds_cfg: dict) -> StructuredDotWorldConfig:
    default_disp = int(ds_cfg["max_displacement"])
    groups: list[DotGroup] = []
    offset = 0
    for g in ds_cfg["groups"]:
        n = int(g["num_dots"])
        groups.append(DotGroup(
            motion_type=MOTION_MAP[g["motion"]],
            num_dots=n,
            color_indices=_auto_colors(n, offset),
            max_displacement=int(g.get("max_displacement", default_disp)),
        ))
        offset += n
    return StructuredDotWorldConfig(
        groups=groups,
        image_size=int(ds_cfg["image_size"]),
        dot_radius=int(ds_cfg["dot_radius"]),
        allow_overlap=bool(ds_cfg.get("allow_overlap", False)),
    )


def build_sprite_config(ds_cfg: dict) -> SpriteWorldConfig:
    """Build a SpriteWorldConfig from a YAML dataset block.

    Reuses datasets.sprite_world.{ControlConfig, Shape} rather than redefining
    the control masks / shape set here.
    """
    raw_theta = ds_cfg.get("max_delta_theta")
    return SpriteWorldConfig(
        control=ControlConfig[str(ds_cfg["control"]).upper()],
        shape=Shape[str(ds_cfg.get("shape", "arrow")).upper()],
        image_size=int(ds_cfg["image_size"]),
        sprite_scale=float(ds_cfg["sprite_scale"]),
        supersample=int(ds_cfg["supersample"]),
        max_delta_xy=float(ds_cfg["max_delta_xy"]),
        max_delta_theta=None if raw_theta is None else float(raw_theta),
    )


def build_world(ds_cfg: dict):
    """Dispatch on ``dataset.world``; return (world_cfg, DatasetClass, action_scale).

    ``action_scale`` normalizes the action vector: a scalar (structured world:
    a single max pixel displacement) or a per-DOF tensor (sprite world: x, y in
    pixels and θ in radians live on different scales).  Both broadcast against
    an (B, action_dim) action tensor.  Defaults to "structured" so existing
    configs are unchanged.
    """
    kind = str(ds_cfg.get("world", "structured")).lower()
    if kind == "structured":
        wcfg = build_structured_config(ds_cfg)
        return wcfg, StructuredDotWorldDataset, float(wcfg.max_displacement)
    if kind == "sprite":
        wcfg = build_sprite_config(ds_cfg)
        return wcfg, SpriteWorldDataset, torch.tensor(wcfg.action_scale, dtype=torch.float32)
    raise ValueError(f"Unknown dataset.world: {kind!r} (expected 'structured' or 'sprite')")


# ─────────────────────────────────────────────────────────────────
#  Data / model construction
# ─────────────────────────────────────────────────────────────────

def materialize(ds) -> TensorDataset:
    """Render a lazy dataset once into stacked tensors."""
    cols = list(zip(*(ds[i] for i in range(len(ds)))))
    return TensorDataset(*(torch.stack(c) for c in cols))


def build_loaders(ds_cfg, world_cfg, DatasetClass, seed, batch_size):
    """Train + eval DataLoaders over once-rendered datasets (disjoint seeds)."""
    def loader(num_samples, ds_seed, shuffle):
        ds = materialize(DatasetClass(world_cfg, num_samples=num_samples, seed=ds_seed))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return (loader(int(ds_cfg["train_samples"]), seed, True),
            loader(int(ds_cfg["eval_samples"]), seed + 999, False))


def build_models(m_cfg, image_size, action_dim, device):
    """Construct (encoder, forward, inverse); order fixed for seed reproducibility."""
    latent_dim, hidden_dim = int(m_cfg["latent_dim"]), int(m_cfg["hidden_dim"])
    encoder = CNNEncoder(latent_dim=latent_dim, image_size=image_size).to(device)
    fwd_model = ForwardModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    inv_model = InverseModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    return encoder, fwd_model, inv_model


# ─────────────────────────────────────────────────────────────────
#  Losses / metrics
# ─────────────────────────────────────────────────────────────────

class Mean:
    """Sample-weighted running mean."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / self.count


def encode_pair(encoder, obs_t, obs_tp1):
    z = encoder(torch.cat([obs_t, obs_tp1], dim=0))
    return z.chunk(2, dim=0)


def compute_losses(models, batch, action_scale, has_action, device):
    """Forward + inverse MSE for one batch — the single definition used by
    both training and eval. An action-free world (sprite 'none' / all-random
    structured) has no inverse signal, hence no anti-collapse pressure."""
    encoder, fwd_model, inv_model = models
    obs_t, action, obs_tp1, _ = batch
    obs_t, obs_tp1, action = obs_t.to(device), obs_tp1.to(device), action.to(device)
    z_t, z_tp1 = encode_pair(encoder, obs_t, obs_tp1)
    a = action / action_scale if has_action else action
    l_fwd = F.mse_loss(fwd_model(z_t, a), z_tp1)
    l_inv = F.mse_loss(inv_model(z_t, z_tp1), a) if has_action else torch.zeros((), device=device)
    return l_fwd, l_inv, z_t, z_tp1


@torch.inference_mode()
def eval_pass(models, loader, action_scale, has_action, lam, device):
    for m in models:
        m.eval()
    fwd, inv = Mean(), Mean()
    z_t_all, z_tp1_all, pos_all, act_all = [], [], [], []
    for batch in loader:
        l_fwd, l_inv, z_t, z_tp1 = compute_losses(models, batch, action_scale, has_action, device)
        fwd.update(l_fwd.item(), z_t.size(0))
        inv.update(l_inv.item(), z_t.size(0))
        z_t_all.append(z_t.cpu()); z_tp1_all.append(z_tp1.cpu())
        pos_all.append(batch[3]); act_all.append(batch[1])
    return {
        "fwd": fwd.value, "inv": inv.value, "total": fwd.value + lam * inv.value,
        "z_t": torch.cat(z_t_all), "z_tp1": torch.cat(z_tp1_all),
        "positions": torch.cat(pos_all), "actions": torch.cat(act_all),
    }


def log_epoch(epoch, epochs, train_hist, eval_metrics):
    msg = (f"epoch {epoch:04d}/{epochs:04d}  "
           f"train: fwd={train_hist['fwd'][-1]:.6f} inv={train_hist['inv'][-1]:.6f}")
    if eval_metrics is not None:
        msg += f"  eval: fwd={eval_metrics['fwd']:.6f} inv={eval_metrics['inv']:.6f}"
    print(msg)


def save_outputs(run_dir, cfg, models, train_hist, eval_hist, snapshots, last_eval):
    encoder, fwd_model, inv_model = models
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.save({
        "encoder": encoder.state_dict(),
        "forward": fwd_model.state_dict(),
        "inverse": inv_model.state_dict(),
    }, run_dir / "model.pt")
    torch.save({"train": train_hist, "eval": eval_hist}, run_dir / "train_history.pt")
    torch.save({
        "snapshots": snapshots,
        "positions": last_eval["positions"],
        "actions": last_eval["actions"],
    }, run_dir / "embeddings.pt")
    print(f"Saved model.pt, config.yaml, train_history.pt, embeddings.pt to {run_dir}")


# ─────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a JEPA-style world model (structured-dot or sprite).")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    ds_cfg, m_cfg, t_cfg = cfg["dataset"], cfg["model"], cfg["training"]
    run_name = cfg["output"]["run_name"]

    seed, epochs, batch_size = int(t_cfg["seed"]), int(t_cfg["epochs"]), int(t_cfg["batch_size"])
    lr, lam = float(t_cfg["lr"]), float(t_cfg["lambda"])
    eval_every = int(t_cfg.get("eval_every", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_cfg, DatasetClass, action_scale = build_world(ds_cfg)
    if isinstance(action_scale, torch.Tensor):
        action_scale = action_scale.to(device)
    action_dim = world_cfg.action_dim
    has_action = action_dim > 0

    run_dir = (config_path.parent / "results" / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config       : {config_path}")
    print(f"Run dir      : {run_dir}")
    print(f"Device       : {device}")
    print(f"Settings     : epochs={epochs} batch_size={batch_size} "
          f"action_dim={action_dim} latent_dim={m_cfg['latent_dim']} lambda={lam}")
    print(world_cfg.describe())

    # Seed before any RNG draw; loaders consume none (numpy-seeded), models do.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Materializing train + eval datasets ...")
    train_loader, eval_loader = build_loaders(ds_cfg, world_cfg, DatasetClass, seed, batch_size)
    models = build_models(m_cfg, world_cfg.image_size, action_dim, device)
    opt = optim.Adam([p for m in models for p in m.parameters()], lr=lr)

    train_hist = {"epoch": [], "fwd": [], "inv": [], "total": []}
    eval_hist = {"epoch": [], "fwd": [], "inv": [], "total": []}
    snapshots: dict[int, dict] = {}
    last_eval = None

    for epoch in range(1, epochs + 1):
        for m in models:
            m.train()
        fwd, inv = Mean(), Mean()
        for batch in train_loader:
            l_fwd, l_inv, z_t, _ = compute_losses(models, batch, action_scale, has_action, device)
            loss = l_fwd + lam * l_inv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            fwd.update(l_fwd.item(), z_t.size(0))
            inv.update(l_inv.item(), z_t.size(0))

        train_hist["epoch"].append(epoch)
        train_hist["fwd"].append(fwd.value)
        train_hist["inv"].append(inv.value)
        train_hist["total"].append(fwd.value + lam * inv.value)

        # The final epoch always evaluates, so last_eval is set before saving.
        if epoch % eval_every == 0 or epoch == epochs:
            last_eval = eval_pass(models, eval_loader, action_scale, has_action, lam, device)
            eval_hist["epoch"].append(epoch)
            for k in ("fwd", "inv", "total"):
                eval_hist[k].append(last_eval[k])
            snapshots[epoch] = {"z_t": last_eval["z_t"], "z_tp1": last_eval["z_tp1"]}
            log_epoch(epoch, epochs, train_hist, last_eval)
        elif epoch <= 5 or epoch % 100 == 0:
            log_epoch(epoch, epochs, train_hist, None)

    save_outputs(run_dir, cfg, models, train_hist, eval_hist, snapshots, last_eval)


if __name__ == "__main__":
    main()
