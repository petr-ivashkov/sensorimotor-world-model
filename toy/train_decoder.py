from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from train import Mean, build_world, load_config
from models import CNNDecoder, CNNEncoder


# ─────────────────────────────────────────────────────────────────
#  Data / model construction
# ─────────────────────────────────────────────────────────────────

def materialize_frames(ds) -> TensorDataset:
    """Render a lazy transition dataset, keeping only obs_t."""
    obs = [ds[i][0] for i in range(len(ds))]
    return TensorDataset(torch.stack(obs))


def build_loaders(ds_cfg, world_cfg, DatasetClass, seed, batch_size):
    """Train + eval obs_t DataLoaders, reusing the encoder run's disjoint seeds."""
    def loader(num_samples, ds_seed, shuffle):
        ds = materialize_frames(DatasetClass(world_cfg, num_samples=num_samples, seed=ds_seed))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return (loader(int(ds_cfg["train_samples"]), seed, True),
            loader(int(ds_cfg["eval_samples"]), seed + 999, False))


def load_frozen_encoder(run_dir, latent_dim, image_size, device):
    """Construct the encoder, load its weights, and freeze it."""
    encoder = CNNEncoder(latent_dim=latent_dim, image_size=image_size).to(device)
    encoder.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["encoder"])
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


# ─────────────────────────────────────────────────────────────────
#  Losses / metrics
# ─────────────────────────────────────────────────────────────────

def compute_losses(encoder, decoder, obs, weight_beta, device):
    """Reconstruction losses for one batch — the single definition used by both
    training and eval. Returns the foreground-weighted L2 (train objective) and
    the plain per-pixel MSE (comparable metric logged for both splits)."""
    obs = obs.to(device)
    with torch.no_grad():
        z = encoder(obs)
    recon = decoder(z)
    weight = 1.0 + weight_beta * (1.0 - obs)   # upweight the sparse sprite pixels
    weighted = (weight * (recon - obs) ** 2).mean()
    plain = F.mse_loss(recon.detach(), obs)
    return weighted, plain


@torch.inference_mode()
def eval_pass(encoder, decoder, loader, weight_beta, device):
    decoder.eval()
    mse = Mean()
    for (obs,) in loader:
        _, plain = compute_losses(encoder, decoder, obs, weight_beta, device)
        mse.update(plain.item(), obs.numel())
    return mse.value


def log_epoch(epoch, epochs, train_mse, eval_mse=None):
    msg = f"epoch {epoch:04d}/{epochs:04d}  train_mse={train_mse:.6f}"
    if eval_mse is not None:
        msg += f"  eval_mse={eval_mse:.6f}"
    print(msg)


def save_outputs(out_dir, decoder, history):
    torch.save(decoder.state_dict(), out_dir / "decoder.pt")
    torch.save(history, out_dir / "decoder_history.pt")
    print(f"Saved decoder.pt, decoder_history.pt to {out_dir}")


# ─────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a probe decoder d_ξ: z_t -> ô_t on a frozen encoder.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    d_cfg = cfg["decoder"]
    encoder_run = cfg["encoder_run"]
    run_name = cfg.get("output", {}).get("run_name", encoder_run)

    epochs, batch_size = int(d_cfg["epochs"]), int(d_cfg["batch_size"])
    lr, seed = float(d_cfg["lr"]), int(d_cfg["seed"])
    eval_every = int(d_cfg.get("eval_every", 5))
    weight_beta = float(d_cfg.get("weight_beta", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The frozen encoder and the world it was trained on are the single source of
    # truth — rebuild the dataset from the encoder run's saved config.
    run_dir = (config_path.parent / "results" / encoder_run).resolve()
    enc_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ds_cfg, m_cfg, t_cfg = enc_cfg["dataset"], enc_cfg["model"], enc_cfg["training"]
    world_cfg, DatasetClass, _ = build_world(ds_cfg)
    latent_dim = int(m_cfg["latent_dim"])
    image_size = world_cfg.image_size
    data_seed = int(t_cfg["seed"])

    out_dir = (config_path.parent / "results" / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config       : {config_path}")
    print(f"Encoder run  : {encoder_run}  ({run_dir})")
    print(f"Out dir      : {out_dir}")
    print(f"Device       : {device}")
    print(f"Settings     : epochs={epochs} batch_size={batch_size} lr={lr} "
          f"latent_dim={latent_dim} weight_beta={weight_beta}")
    print(world_cfg.describe())

    # Seed before any RNG draw; loaders consume none, the decoder's init does.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Materializing train + eval observations ...")
    train_loader, eval_loader = build_loaders(ds_cfg, world_cfg, DatasetClass, data_seed, batch_size)
    encoder = load_frozen_encoder(run_dir, latent_dim, image_size, device)
    decoder = CNNDecoder(latent_dim=latent_dim, image_size=image_size).to(device)
    opt = optim.Adam(decoder.parameters(), lr=lr)

    history = {"train": {"epoch": [], "mse": []}, "eval": {"epoch": [], "mse": []}}

    for epoch in range(1, epochs + 1):
        decoder.train()
        mse = Mean()
        for (obs,) in train_loader:
            weighted, plain = compute_losses(encoder, decoder, obs, weight_beta, device)
            opt.zero_grad(set_to_none=True)
            weighted.backward()
            opt.step()
            mse.update(plain.item(), obs.numel())

        history["train"]["epoch"].append(epoch)
        history["train"]["mse"].append(mse.value)

        if epoch % eval_every == 0 or epoch == epochs:
            ev = eval_pass(encoder, decoder, eval_loader, weight_beta, device)
            history["eval"]["epoch"].append(epoch)
            history["eval"]["mse"].append(ev)
            log_epoch(epoch, epochs, mse.value, ev)
        elif epoch <= 5:
            log_epoch(epoch, epochs, mse.value)

    save_outputs(out_dir, decoder, history)


if __name__ == "__main__":
    main()
