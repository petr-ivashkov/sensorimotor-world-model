"""Train a JEPA-style world model with a *multistep* inverse-dynamics regularizer.

Same models and joint objective as train.py, except the inverse head decodes
the k-step endpoints into the *mean* action over the horizon:

    L = ||g_ϕ(z_t, a_t) − z_{t+1}||² + λ·||h_ψ(z_t, z_{t+k}) − ā_t||²,
    ā_t = (1/k)·Σ_{i=0}^{k-1} a_{t+i}

The forward loss stays single-step (first transition of each segment); only
the inverse horizon widens.  The mean action is the target — not the action
sequence — because the endpoints (z_t, z_{t+k}) do not identify the individual
actions, only (roughly) their sum.

``training.horizon_k = 1`` reproduces train.py exactly: the segment datasets
draw the same RNG sequence as the transition datasets, the losses reduce to
train.compute_losses, and the update sequence is identical (verified in
tests/test_multistep_inverse.py).  Run directories use the same layout as
train.py (model.pt, config.yaml, train_history.pt, embeddings.pt), so the
existing notebooks, train_decoder.py, and eval_goal_reaching.py work on them
unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from datasets.structured_dot_world import (
    StructuredDotWorldDataset,
    StructuredDotWorldSegmentDataset,
)
from datasets.sprite_world import (
    SpriteWorldDataset,
    SpriteWorldSegmentDataset,
)
from train import (
    Mean,
    build_models,
    build_world,
    encode_pair,
    load_config,
    log_epoch,
    materialize,
)


# Transition dataset (train.build_world) → its k-step segment counterpart.
SEGMENT_DATASETS = {
    StructuredDotWorldDataset: StructuredDotWorldSegmentDataset,
    SpriteWorldDataset: SpriteWorldSegmentDataset,
}


# ─────────────────────────────────────────────────────────────────
#  Data construction
# ─────────────────────────────────────────────────────────────────

def build_loaders(ds_cfg, world_cfg, DatasetClass, seed, batch_size, horizon_k):
    """Train + eval DataLoaders over once-rendered k-step segments (disjoint seeds)."""
    SegmentClass = SEGMENT_DATASETS[DatasetClass]

    def loader(num_samples, ds_seed, shuffle):
        ds = materialize(SegmentClass(
            world_cfg, num_samples=num_samples, seed=ds_seed, num_steps=horizon_k,
        ))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (loader(int(ds_cfg["train_samples"]), seed, True),
            loader(int(ds_cfg["eval_samples"]), seed + 999, False))


# ─────────────────────────────────────────────────────────────────
#  Losses / metrics
# ─────────────────────────────────────────────────────────────────

def compute_losses(models, batch, action_scale, has_action, device, horizon_k):
    """Forward + multistep inverse MSE for one batch — the single definition
    used by both training and eval.  For horizon_k == 1 this computes exactly
    train.compute_losses (obs_tpk ≡ obs_tp1, ā_t ≡ a_t)."""
    encoder, fwd_model, inv_model = models
    obs_t, action, obs_tp1, obs_tpk, action_mean, _ = batch
    obs_t, obs_tp1, action = obs_t.to(device), obs_tp1.to(device), action.to(device)
    a = action / action_scale if has_action else action
    if horizon_k == 1:
        z_t, z_tp1 = encode_pair(encoder, obs_t, obs_tp1)
        z_tpk, a_mean = z_tp1, a
    else:
        obs_tpk, action_mean = obs_tpk.to(device), action_mean.to(device)
        z = encoder(torch.cat([obs_t, obs_tp1, obs_tpk], dim=0))
        z_t, z_tp1, z_tpk = z.chunk(3, dim=0)
        a_mean = action_mean / action_scale if has_action else action_mean
    l_fwd = F.mse_loss(fwd_model(z_t, a), z_tp1)
    l_inv = F.mse_loss(inv_model(z_t, z_tpk), a_mean) if has_action else torch.zeros((), device=device)
    return l_fwd, l_inv, z_t, z_tp1, z_tpk


@torch.inference_mode()
def eval_pass(models, loader, action_scale, has_action, lam, device, horizon_k):
    for m in models:
        m.eval()
    fwd, inv = Mean(), Mean()
    z_t_all, z_tp1_all, z_tpk_all = [], [], []
    pos_all, act_all, act_mean_all = [], [], []
    for batch in loader:
        l_fwd, l_inv, z_t, z_tp1, z_tpk = compute_losses(
            models, batch, action_scale, has_action, device, horizon_k)
        fwd.update(l_fwd.item(), z_t.size(0))
        inv.update(l_inv.item(), z_t.size(0))
        z_t_all.append(z_t.cpu()); z_tp1_all.append(z_tp1.cpu()); z_tpk_all.append(z_tpk.cpu())
        pos_all.append(batch[5]); act_all.append(batch[1]); act_mean_all.append(batch[4])
    return {
        "fwd": fwd.value, "inv": inv.value, "total": fwd.value + lam * inv.value,
        "z_t": torch.cat(z_t_all), "z_tp1": torch.cat(z_tp1_all), "z_tpk": torch.cat(z_tpk_all),
        "positions": torch.cat(pos_all), "actions": torch.cat(act_all),
        "actions_mean": torch.cat(act_mean_all),
    }


def save_outputs(run_dir, cfg, models, train_hist, eval_hist, snapshots, last_eval):
    """train.save_outputs layout, plus the multistep fields (z_tpk, actions_mean)."""
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
        "actions_mean": last_eval["actions_mean"],
    }, run_dir / "embeddings.pt")
    print(f"Saved model.pt, config.yaml, train_history.pt, embeddings.pt to {run_dir}")


# ─────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a JEPA-style world model with a multistep (horizon-k) "
                    "inverse-dynamics regularizer (structured-dot or sprite).")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    ds_cfg, m_cfg, t_cfg = cfg["dataset"], cfg["model"], cfg["training"]
    run_name = cfg["output"]["run_name"]

    seed, epochs, batch_size = int(t_cfg["seed"]), int(t_cfg["epochs"]), int(t_cfg["batch_size"])
    lr, lam = float(t_cfg["lr"]), float(t_cfg["lambda"])
    eval_every = int(t_cfg.get("eval_every", 10))
    horizon_k = int(t_cfg.get("horizon_k", 1))

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
          f"action_dim={action_dim} latent_dim={m_cfg['latent_dim']} lambda={lam} "
          f"horizon_k={horizon_k}")
    print(world_cfg.describe())

    # Seed before any RNG draw; loaders consume none (numpy-seeded), models do.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Materializing train + eval segment datasets ...")
    train_loader, eval_loader = build_loaders(
        ds_cfg, world_cfg, DatasetClass, seed, batch_size, horizon_k)
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
            l_fwd, l_inv, z_t, _, _ = compute_losses(
                models, batch, action_scale, has_action, device, horizon_k)
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
            last_eval = eval_pass(models, eval_loader, action_scale, has_action,
                                  lam, device, horizon_k)
            eval_hist["epoch"].append(epoch)
            for k in ("fwd", "inv", "total"):
                eval_hist[k].append(last_eval[k])
            snapshots[epoch] = {"z_t": last_eval["z_t"], "z_tp1": last_eval["z_tp1"],
                                "z_tpk": last_eval["z_tpk"]}
            log_epoch(epoch, epochs, train_hist, last_eval)
        elif epoch <= 5 or epoch % 100 == 0:
            log_epoch(epoch, epochs, train_hist, None)

    save_outputs(run_dir, cfg, models, train_hist, eval_hist, snapshots, last_eval)


if __name__ == "__main__":
    main()
