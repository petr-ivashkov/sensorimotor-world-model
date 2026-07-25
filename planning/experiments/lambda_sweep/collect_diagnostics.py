#!/usr/bin/env python3
"""Compute held-out latent and prediction diagnostics for one trained run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval import load_jepa_from_run
from utils import compute_embedding_validation_stats


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def transition_errors(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    actions: torch.Tensor,
    device: str,
    batch_size: int,
) -> tuple[float, float, int]:
    if embeddings.ndim != 3 or actions.ndim != 3:
        raise ValueError("Expected embeddings and actions with shape [N, T, D]")
    if embeddings.shape[:2] != actions.shape[:2]:
        raise ValueError("Embedding and action sequence dimensions do not match")
    if embeddings.shape[1] < 2:
        raise ValueError("At least two time steps are required")

    z_t = embeddings[:, :-1].reshape(-1, embeddings.shape[-1])
    z_tp1 = embeddings[:, 1:].reshape(-1, embeddings.shape[-1])
    action = torch.nan_to_num(
        actions[:, :-1].reshape(-1, actions.shape[-1]), 0.0
    )

    forward_sum = 0.0
    inverse_sum = 0.0
    forward_elements = 0
    inverse_elements = 0

    with torch.inference_mode():
        for start in range(0, z_t.shape[0], batch_size):
            stop = min(start + batch_size, z_t.shape[0])
            current = z_t[start:stop].to(device)
            target = z_tp1[start:stop].to(device)
            target_action = action[start:stop].to(device)

            action_embedding = model.action_encoder(target_action.unsqueeze(1))
            predicted = model.predict(current.unsqueeze(1), action_embedding).squeeze(1)
            predicted_action = model.predict_action(current, target)

            forward_sum += float(
                F.mse_loss(predicted, target, reduction="sum").item()
            )
            inverse_sum += float(
                F.mse_loss(predicted_action, target_action, reduction="sum").item()
            )
            forward_elements += target.numel()
            inverse_elements += target_action.numel()

    return (
        forward_sum / forward_elements,
        inverse_sum / inverse_elements,
        int(z_t.shape[0]),
    )


def collect(run_dir: Path, device: str, batch_size: int) -> dict[str, object]:
    config_path = run_dir / "config.yaml"
    embeddings_path = run_dir / "final_embeddings.pt"
    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    for path in (config_path, embeddings_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = OmegaConf.load(config_path)
    if int(cfg.wm.get("history_size", 1)) != 1:
        raise ValueError("Lambda sweep diagnostics require wm.history_size=1")

    payload = torch.load(embeddings_path, map_location="cpu", weights_only=True)
    embeddings = payload["emb"].float()
    actions = payload["action"].float()
    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    representation = compute_embedding_validation_stats(emb=flat_embeddings)
    mean_per_dim_variance = float(
        flat_embeddings.var(dim=0, unbiased=False).mean().item()
    )

    model, _ = load_jepa_from_run(run_dir, device=device)
    forward_mse, inverse_mse, num_transitions = transition_errors(
        model=model,
        embeddings=embeddings,
        actions=actions,
        device=device,
        batch_size=batch_size,
    )

    sweep_cfg = cfg.get("lambda_sweep", {})
    return {
        "run_name": run_dir.name,
        "environment": str(sweep_cfg.get("environment", "")),
        "lambda": float(cfg.loss.inverse.weight),
        "seed": int(cfg.seed),
        "history": int(cfg.wm.history_size),
        "source": "final_embeddings.pt",
        "num_sequences": int(embeddings.shape[0]),
        "num_embeddings": int(flat_embeddings.shape[0]),
        "num_transitions": num_transitions,
        "metrics": {
            "effective_rank": float(representation["effective_rank"]),
            "mean_per_dim_variance": mean_per_dim_variance,
            "mean_latent_vector_length": float(representation["mean_norm"]),
            "inverse_mse": inverse_mse,
            "forward_mse": forward_mse,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "diagnostics.json"
    result = collect(
        run_dir=run_dir,
        device=choose_device(args.device),
        batch_size=args.batch_size,
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote diagnostics to {output}")


if __name__ == "__main__":
    main()
