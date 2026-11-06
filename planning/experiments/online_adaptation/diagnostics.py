"""Fixed-anchor and episode-separated diagnostics for the H=1 Push-T pilot.

Loaders must use disjoint fit/test episodes and deterministic, fixed windows.
Each test window needs ``max(horizons) + 1`` observations. Action token ``t``
contains the five outgoing, source-normalized 2D actions from observation t.
The terminal action is never consumed. These are diagnostics, not evidence of
control improvement or a standalone test for representation collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_EPS = 1e-12


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Centered linear CKA, invariant to isotropic scaling and rotation.

    Return NaN for a constant representation, where CKA is undefined.
    """
    x, y = x.double(), y.double()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be matrices with the same sample count")
    x = x - x.mean(0)
    y = y - y.mean(0)
    denominator = torch.linalg.norm(x.T @ x) * torch.linalg.norm(y.T @ y)
    if denominator <= _EPS:
        return float("nan")
    return float((torch.linalg.norm(x.T @ y).square() / denominator).clamp(0, 1))


def physical_targets(state: torch.Tensor) -> torch.Tensor:
    """Agent xy, block xy, sin(theta), cos(theta); raw Push-T angle is index 4."""
    if state.shape[-1] != 7:
        raise ValueError("Expected raw seven-component Push-T state")
    return torch.cat((state[..., :4], state[..., 4:5].sin(), state[..., 4:5].cos()), -1)


def circular_mae(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute wrapped angular error, in radians."""
    delta = predicted - target
    return float(torch.atan2(delta.sin(), delta.cos()).abs().mean())


def _physical_errors(decoded: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    delta = decoded - targets
    # Position RMSE is sqrt(mean(squared Euclidean error)), in simulator pixels.
    return {
        "agent_position_rmse": float(delta[..., :2].square().sum(-1).mean().sqrt()),
        "block_position_rmse": float(delta[..., 2:4].square().sum(-1).mean().sqrt()),
        "block_angle_mae_rad": circular_mae(
            torch.atan2(decoded[..., 4], decoded[..., 5]),
            torch.atan2(targets[..., 4], targets[..., 5]),
        ),
    }


def _fit_ridge(features: torch.Tensor, targets: torch.Tensor, alpha: float):
    """Fit only caller-provided fitting episodes, including all preprocessing."""
    x, y = features.double(), targets.double()
    mean = x.mean(0)
    scale = x.std(0, correction=0).clamp_min(1e-8)
    x = (x - mean) / scale
    y_mean = y.mean(0)
    centered_y = y - y_mean
    if alpha == 0:
        weights = torch.linalg.lstsq(x, centered_y).solution
    elif x.shape[0] < x.shape[1]:
        kernel = x @ x.T + alpha * torch.eye(x.shape[0], dtype=x.dtype)
        weights = x.T @ torch.linalg.solve(kernel, centered_y)
    else:
        gram = x.T @ x + alpha * torch.eye(x.shape[1], dtype=x.dtype)
        weights = torch.linalg.solve(gram, x.T @ centered_y)

    def decode(values: torch.Tensor):
        return ((values.double() - mean) / scale) @ weights + y_mean

    return decode


def _episode_ids(batch: dict) -> set:
    if "ep_idx" not in batch:
        return set()
    values = batch["ep_idx"]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return set(np.asarray(values).reshape(-1).tolist())


def _batches(loader, max_batches):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def _encode(model, pixels: torch.Tensor, device) -> torch.Tensor:
    info = model.encode({"pixels": pixels.to(device)})
    return info["emb"]


def _validate_batch(batch: dict, min_time: int):
    pixels, state = batch["pixels"], batch["state"]
    if pixels.ndim != 5 or state.ndim != 3 or pixels.shape[:2] != state.shape[:2]:
        raise ValueError("Expected pixels B,T,C,H,W and matching state B,T,7")
    if pixels.shape[1] < min_time:
        raise ValueError(f"Window needs at least {min_time} observations")
    if state.shape[-1] != 7 or not torch.isfinite(state).all():
        raise ValueError("Expected finite raw seven-component Push-T states")


def _representation_metrics(current: torch.Tensor, baseline: torch.Tensor):
    current = current.double()
    centered = current - current.mean(0)
    singular_values = torch.linalg.svdvals(centered)
    mass = singular_values.sum()
    effective_rank = 0.0
    if mass > _EPS:
        probabilities = singular_values / mass
        positive = probabilities > 0
        effective_rank = float((-(probabilities[positive] * probabilities[positive].log()).sum()).exp())
    metrics = {
        "representation.anchor_count": int(current.shape[0]),
        "representation.std_mean": float(current.std(0, correction=0).mean()),
        "representation.effective_rank": effective_rank,
        "representation.linear_cka": linear_cka(current, baseline),
    }
    if current.shape == baseline.shape:
        # Coordinate drift intentionally remains sensitive to rotation/translation.
        denominator = baseline.double().square().mean().sqrt().clamp_min(_EPS)
        metrics["representation.relative_coordinate_drift"] = float(
            (current - baseline).square().mean().sqrt() / denominator
        )
    return metrics


def evaluate_diagnostics(
    model,
    fit_loader,
    test_loader,
    *,
    reference: dict | None = None,
    horizons=(1, 3, 5),
    ridge_alpha=1.0,
    device="cuda",
    max_batches=None,
) -> tuple[dict, dict]:
    """Evaluate a snapshot against fixed held-out windows and baseline anchors.

    A fresh ridge probe is fitted to t=0 embeddings from ``fit_loader`` only;
    its feature scaling also uses only that split. The caller must separate
    episodes, not randomly split frames/windows. Overlap is rejected when both
    loaders expose comparable ``ep_idx`` identifiers. IDs should retain their
    original dataset meaning after subsetting.

    ``reference=None`` captures t=0 test pixels, raw states, and baseline
    embeddings as cloned CPU tensors. With a reference, its stored pixels are
    re-encoded for drift/CKA and the exact supplied dictionary is returned
    unchanged. Save it with torch.save to keep an auditable anchor set.

    Action errors use source-normalized coordinates. Latent normalized MSE
    divides raw MSE by the held-out target latent variance at that horizon;
    report it alongside physical errors, probe errors, rank, and variance.
    Rank uses entropy of singular values. Undefined constant-space CKA is NaN.
    All original module train/eval flags are restored, including frozen modules.
    """
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be nonnegative")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    if int(getattr(model.predictor, "num_frames", 1)) != 1:
        raise ValueError("Online pilot diagnostics require predictor history H=1")

    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            return _evaluate(
                model, fit_loader, test_loader, reference, horizons,
                float(ridge_alpha), torch.device(device), max_batches,
            )
    finally:
        for module, training in modes:
            module.training = training


def _evaluate(model, fit_loader, test_loader, reference, horizons, alpha, device, max_batches):
    fit_features, fit_states, fit_ids = [], [], set()
    for batch in _batches(fit_loader, max_batches):
        _validate_batch(batch, 1)
        fit_ids.update(_episode_ids(batch))
        fit_features.append(_encode(model, batch["pixels"][:, :1], device)[:, 0].float().cpu())
        fit_states.append(batch["state"][:, 0].float().cpu())
    if not fit_features:
        raise ValueError("fit_loader is empty")
    fit_features, fit_states = torch.cat(fit_features), torch.cat(fit_states)
    if fit_features.shape[0] < 2:
        raise ValueError("Need at least two fitting samples")
    decode = _fit_ridge(fit_features, physical_targets(fit_states), alpha)

    anchor_pixels, anchor_states, anchor_features = [], [], []
    actual, predicted, future_states = ({h: [] for h in horizons} for _ in range(3))
    initial_features, initial_states, inverse_errors = [], [], []
    test_ids = set()
    has_inverse = getattr(model, "inverse_model", None) is not None
    for batch in _batches(test_loader, max_batches):
        _validate_batch(batch, max(horizons) + 1)
        test_ids.update(_episode_ids(batch))
        if fit_ids & test_ids:
            raise ValueError("Probe fit/test episode IDs overlap; split whole episodes")
        embeddings = _encode(model, batch["pixels"], device)
        actions = batch["action"]
        if actions.ndim != 3 or actions.shape[0] != embeddings.shape[0] or actions.shape[-1] != 10:
            raise ValueError("Expected source-normalized action tokens B,T,10")
        if actions.shape[1] < max(horizons) or not torch.isfinite(actions[:, :max(horizons)]).all():
            raise ValueError("Rollout actions must be finite and cover every requested transition")
        initial_features.append(embeddings[:, 0].float().cpu())
        initial_states.append(batch["state"][:, 0].float().cpu())
        if reference is None:
            anchor_pixels.append(batch["pixels"][:, :1].detach().cpu().clone())
            anchor_states.append(batch["state"][:, 0].detach().cpu().clone())
            anchor_features.append(embeddings[:, 0].float().cpu())
        if has_inverse:
            inverse = model.predict_action(embeddings[:, 0], embeddings[:, 1])
            inverse_errors.append((inverse - actions[:, 0].to(device)).float().cpu())
        latent = embeddings[:, :1]
        for step in range(max(horizons)):
            action = actions[:, step:step + 1].to(device)
            latent = model.predict(latent, model.action_encoder(action))[:, -1:]
            horizon = step + 1
            if horizon in actual:
                predicted[horizon].append(latent[:, 0].float().cpu())
                actual[horizon].append(embeddings[:, horizon].float().cpu())
                future_states[horizon].append(batch["state"][:, horizon].float().cpu())
    if not initial_features:
        raise ValueError("test_loader is empty")
    initial_features, initial_states = torch.cat(initial_features), torch.cat(initial_states)
    if reference is None:
        reference = {
            "pixels": torch.cat(anchor_pixels).clone(),
            "state": torch.cat(anchor_states).clone(),
            "embeddings": torch.cat(anchor_features).clone(),
            "episode_ids": sorted(test_ids, key=str),
        }
        current_anchors = reference["embeddings"]
    else:
        pixels = reference["pixels"]
        baseline = reference["embeddings"]
        if pixels.ndim != 5 or baseline.ndim != 2 or pixels.shape[0] != baseline.shape[0]:
            raise ValueError("Reference pixels/embeddings have incompatible anchor counts")
        if pixels.shape[0] == 0:
            raise ValueError("Reference anchors are empty")
        chunk_size = 32
        current_anchors = torch.cat([
            _encode(model, pixels[start:start + chunk_size], device)[:, 0].float().cpu()
            for start in range(0, pixels.shape[0], chunk_size)
        ])
    metrics: dict[str, Any] = _representation_metrics(current_anchors, reference["embeddings"])
    metrics.update({
        "probe.fit_samples": int(fit_features.shape[0]),
        "probe.test_samples": int(initial_features.shape[0]),
        "probe.fit_episode_count": len(fit_ids),
        "probe.test_episode_count": len(test_ids),
        "probe.ridge_alpha": alpha,
    })
    metrics.update({
        "observed." + key: value for key, value in
        _physical_errors(decode(initial_features), physical_targets(initial_states)).items()
    })
    if inverse_errors:
        error = torch.cat(inverse_errors)
        metrics["inverse.action_mse"] = float(error.square().mean())
        metrics["inverse.action_mae"] = float(error.abs().mean())
    for horizon in horizons:
        truth = torch.cat(actual[horizon])
        prediction = torch.cat(predicted[horizon])
        targets = physical_targets(torch.cat(future_states[horizon]))
        prefix = f"horizon_{horizon}."
        mse = float((prediction - truth).square().mean())
        variance = float(truth.var(0, correction=0).mean())
        metrics[prefix + "latent_mse"] = mse
        metrics[prefix + "target_latent_variance"] = variance
        metrics[prefix + "normalized_latent_mse"] = mse / max(variance, _EPS)
        for name, values in (
            ("prediction", decode(prediction)),
            ("observed", decode(truth)),
            ("persistence", physical_targets(initial_states)),
        ):
            metrics.update({
                prefix + name + "." + key: value
                for key, value in _physical_errors(values, targets).items()
            })
    return metrics, reference
