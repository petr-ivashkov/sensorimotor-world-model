"""Train a JEPA-style world model with a *multistep* inverse-dynamics regularizer.

Same models and joint objective as train.py, except the inverse head decodes
k-step endpoints into the FIRST action of the segment, with the horizon k
sampled per example — the multi-step inverse construction of Lamb et al.
(arXiv 2207.08229, "Guaranteed Discovery of Control-Endogenous Latent State
with Multi-Step Inverse Models"):

    L = ||g_ϕ(z_t, a_t) − z_{t+1}||² + λ·||h_ψ(z_t, z_{t+k}, k) − a_t||²,
    k ~ Uniform{k_min .. k_max}  per training example

The forward loss stays single-step (first transition of each segment). The
head is conditioned on k explicitly (one-hot through the first layer — see
models.MultistepInverseModel for why a horizon-agnostic head cannot represent
the per-k optimum).

Caveat, flagged not fixed: Lamb et al.'s identifiability result assumes finite
actions and deterministic endogenous dynamics. Here actions are continuous, so
(z_t, z_{t+k}) does not determine a_t — many action sequences share endpoints —
and an MSE head regresses the conditional mean E[a_t | z_t, z_{t+k}, k]. We
test this empirically as a regularizer; to make degradation at larger horizons
visible rather than hidden in an aggregate, the eval pass logs the inverse
loss PER HORIZON k (``eval_hist["inv_per_k"]``, and per-sample horizons in
embeddings.pt).

``horizon_k_max = 1`` reproduces train.py exactly: the segment datasets draw
the same RNG sequence as the transition datasets, the models are built by
train.build_models (plain InverseModel, no k input), and the update sequence
is identical (verified bitwise in tests/test_multistep_inverse.py). Run
directories use the same layout as train.py (model.pt, config.yaml,
train_history.pt, embeddings.pt), so train_decoder.py, the notebooks, and
eval_goal_reaching.py work on them unchanged.
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
from models import CNNEncoder, ForwardModel, MultistepInverseModel
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
#  Data / model construction
# ─────────────────────────────────────────────────────────────────

def build_loaders(ds_cfg, world_cfg, DatasetClass, seed, batch_size, k_min, k_max):
    """Train + eval DataLoaders over once-rendered random-k segments (disjoint seeds)."""
    SegmentClass = SEGMENT_DATASETS[DatasetClass]

    def loader(num_samples, ds_seed, shuffle):
        ds = materialize(SegmentClass(
            world_cfg, num_samples=num_samples, seed=ds_seed, k_min=k_min, k_max=k_max,
        ))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (loader(int(ds_cfg["train_samples"]), seed, True),
            loader(int(ds_cfg["eval_samples"]), seed + 999, False))


def build_multistep_models(m_cfg, image_size, action_dim, k_max, device):
    """Construct (encoder, forward, inverse).

    For k_max == 1 this defers to train.build_models — same classes, same
    construction order, same init RNG draws — so the whole run is bitwise
    train.py. For k_max > 1 the inverse head is the k-conditioned
    MultistepInverseModel (encoder/forward unchanged, same order).
    """
    if k_max == 1:
        return build_models(m_cfg, image_size, action_dim, device)
    latent_dim, hidden_dim = int(m_cfg["latent_dim"]), int(m_cfg["hidden_dim"])
    encoder = CNNEncoder(latent_dim=latent_dim, image_size=image_size).to(device)
    fwd_model = ForwardModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    inv_model = MultistepInverseModel(latent_dim=latent_dim, action_dim=action_dim,
                                      hidden_dim=hidden_dim, k_max=k_max).to(device)
    return encoder, fwd_model, inv_model


# ─────────────────────────────────────────────────────────────────
#  Losses / metrics
# ─────────────────────────────────────────────────────────────────

def compute_losses(models, batch, action_scale, has_action, device, k_max):
    """Forward + multistep first-action inverse MSE for one batch — the single
    definition used by both training and eval. For k_max == 1 this computes
    exactly train.compute_losses (obs_tpk ≡ obs_tp1, plain inverse head)."""
    encoder, fwd_model, inv_model = models
    obs_t, action, obs_tp1, obs_tpk, ks, _ = batch
    obs_t, obs_tp1, action = obs_t.to(device), obs_tp1.to(device), action.to(device)
    a = action / action_scale if has_action else action
    if k_max == 1:
        z_t, z_tp1 = encode_pair(encoder, obs_t, obs_tp1)
        z_tpk = z_tp1
        a_hat = inv_model(z_t, z_tpk) if has_action else None
    else:
        obs_tpk, ks = obs_tpk.to(device), ks.to(device)
        z = encoder(torch.cat([obs_t, obs_tp1, obs_tpk], dim=0))
        z_t, z_tp1, z_tpk = z.chunk(3, dim=0)
        a_hat = inv_model(z_t, z_tpk, ks) if has_action else None
    l_fwd = F.mse_loss(fwd_model(z_t, a), z_tp1)
    l_inv = F.mse_loss(a_hat, a) if has_action else torch.zeros((), device=device)
    return l_fwd, l_inv, a_hat, z_t, z_tp1, z_tpk


@torch.inference_mode()
def eval_pass(models, loader, action_scale, has_action, lam, device, k_max):
    for m in models:
        m.eval()
    fwd, inv = Mean(), Mean()
    inv_per_k = {k: Mean() for k in range(1, k_max + 1)}
    z_t_all, z_tp1_all, z_tpk_all = [], [], []
    pos_all, act_all, k_all = [], [], []
    for batch in loader:
        l_fwd, l_inv, a_hat, z_t, z_tp1, z_tpk = compute_losses(
            models, batch, action_scale, has_action, device, k_max)
        fwd.update(l_fwd.item(), z_t.size(0))
        inv.update(l_inv.item(), z_t.size(0))
        if has_action:
            # Per-horizon inverse loss — the multimodality instrumentation.
            a = (batch[1].to(device) / action_scale)
            per_sample = ((a_hat - a) ** 2).mean(dim=1)
            for k in inv_per_k:
                mask = batch[4].to(device) == k
                if mask.any():
                    inv_per_k[k].update(per_sample[mask].mean().item(), int(mask.sum()))
        z_t_all.append(z_t.cpu()); z_tp1_all.append(z_tp1.cpu()); z_tpk_all.append(z_tpk.cpu())
        pos_all.append(batch[5]); act_all.append(batch[1]); k_all.append(batch[4])
    return {
        "fwd": fwd.value, "inv": inv.value, "total": fwd.value + lam * inv.value,
        "inv_per_k": {k: m.value for k, m in inv_per_k.items() if m.count > 0},
        "z_t": torch.cat(z_t_all), "z_tp1": torch.cat(z_tp1_all), "z_tpk": torch.cat(z_tpk_all),
        "positions": torch.cat(pos_all), "actions": torch.cat(act_all),
        "ks": torch.cat(k_all),
    }


def save_outputs(run_dir, cfg, models, train_hist, eval_hist, snapshots, last_eval):
    """train.save_outputs layout, plus the multistep fields (z_tpk, ks)."""
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
        "ks": last_eval["ks"],
    }, run_dir / "embeddings.pt")
    print(f"Saved model.pt, config.yaml, train_history.pt, embeddings.pt to {run_dir}")


# ─────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a JEPA-style world model with a multistep first-action "
                    "inverse regularizer, k ~ U{k_min..k_max} per example "
                    "(structured-dot or sprite).")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    ds_cfg, m_cfg, t_cfg = cfg["dataset"], cfg["model"], cfg["training"]
    run_name = cfg["output"]["run_name"]

    seed, epochs, batch_size = int(t_cfg["seed"]), int(t_cfg["epochs"]), int(t_cfg["batch_size"])
    lr, lam = float(t_cfg["lr"]), float(t_cfg["lambda"])
    eval_every = int(t_cfg.get("eval_every", 10))
    k_min = int(t_cfg.get("horizon_k_min", 1))
    k_max = int(t_cfg.get("horizon_k_max", 1))
    if not 1 <= k_min <= k_max:
        raise ValueError(f"Need 1 <= horizon_k_min <= horizon_k_max, got ({k_min}, {k_max})")

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
          f"horizon_k=U{{{k_min}..{k_max}}}")
    print(world_cfg.describe())

    # Seed before any RNG draw; loaders consume none (numpy-seeded), models do.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Materializing train + eval segment datasets ...")
    train_loader, eval_loader = build_loaders(
        ds_cfg, world_cfg, DatasetClass, seed, batch_size, k_min, k_max)
    models = build_multistep_models(m_cfg, world_cfg.image_size, action_dim, k_max, device)
    opt = optim.Adam([p for m in models for p in m.parameters()], lr=lr)

    train_hist = {"epoch": [], "fwd": [], "inv": [], "total": []}
    eval_hist = {"epoch": [], "fwd": [], "inv": [], "total": [], "inv_per_k": []}
    snapshots: dict[int, dict] = {}
    last_eval = None

    for epoch in range(1, epochs + 1):
        for m in models:
            m.train()
        fwd, inv = Mean(), Mean()
        for batch in train_loader:
            l_fwd, l_inv, _, z_t, _, _ = compute_losses(
                models, batch, action_scale, has_action, device, k_max)
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
                                  lam, device, k_max)
            eval_hist["epoch"].append(epoch)
            for key in ("fwd", "inv", "total", "inv_per_k"):
                eval_hist[key].append(last_eval[key])
            snapshots[epoch] = {"z_t": last_eval["z_t"], "z_tp1": last_eval["z_tp1"],
                                "z_tpk": last_eval["z_tpk"]}
            log_epoch(epoch, epochs, train_hist, last_eval)
        elif epoch <= 5 or epoch % 100 == 0:
            log_epoch(epoch, epochs, train_hist, None)

    if k_max > 1 and last_eval["inv_per_k"]:
        per_k = "  ".join(f"k={k}: {v:.6f}" for k, v in sorted(last_eval["inv_per_k"].items()))
        print(f"Final eval inverse loss per horizon  {per_k}")
    save_outputs(run_dir, cfg, models, train_hist, eval_hist, snapshots, last_eval)


if __name__ == "__main__":
    main()
