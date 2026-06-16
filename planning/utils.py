"""Utilities adapted from Le-WM."""

import os
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class ResizeCompat:
    """Version-tolerant image resize transform for stable_pretraining pipelines."""

    def __init__(self, size: int, source: str = "image", target: str = "image"):
        import stable_pretraining as spt
        from torchvision.transforms import v2

        class _ResizeTransform(spt.data.transforms.Transform):
            def __init__(self, resize, source, target):
                super().__init__()
                self.resize = resize
                self.source = source
                self.target = target

            def __call__(self, x):
                self.nested_set(x, self.resize(self.nested_get(x, self.source)), self.target)
                return x

        self.transform = _ResizeTransform(v2.Resize(size), source, target)

    def __call__(self, x):
        return self.transform(x)

def get_column_normalizer(dataset, source: str, target: str):
    import stable_pretraining as spt
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = spt.data.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


def load_composed_config(config_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    os.environ.setdefault("REPO_ROOT", str(Path(__file__).resolve().parent))
    config_path = Path(config_path).expanduser().resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_path.parent),
        job_name=config_path.stem,
    ):
        return compose(config_name=config_path.stem)


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


def _flatten_time_pairs(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :-1], x[:, 1:]], dim=1).reshape(-1, x.shape[-1])


def _effective_rank(x: torch.Tensor) -> float:
    x = x.float()
    centered = x - x.mean(dim=0, keepdim=True)
    cov = centered.T @ centered
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
    singular_values = eigvals.sqrt()
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    p = singular_values / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def _ridge_r2(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    alpha: float = 1.0,
) -> float:
    x_train = x_train.float()
    x_test = x_test.float()
    y_train = y_train.float().reshape(-1, 1)
    y_test = y_test.float().reshape(-1, 1)

    x_mean = x_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)
    x_train = x_train - x_mean
    x_test = x_test - x_mean
    y_train = y_train - y_mean

    eye = torch.eye(x_train.shape[1], dtype=x_train.dtype)
    weights = torch.linalg.solve(
        x_train.T @ x_train + alpha * eye,
        x_train.T @ y_train,
    )
    y_pred = x_test @ weights + y_mean

    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean(dim=0, keepdim=True)) ** 2).sum()
    if ss_tot <= 0:
        return float("nan")
    return float((1.0 - ss_res / ss_tot).item())


def _mean_metric(metrics: dict[str, float], names: list[str]) -> float:
    values = [metrics[name] for name in names if name in metrics]
    if not values:
        return float("nan")
    return float(np.mean(values))


def compute_embedding_validation_stats(
    emb: torch.Tensor,
    state: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    alpha: float = 1.0,
    split_seed: int = 0,
) -> dict[str, float]:
    emb = emb.float()
    norms = emb.norm(dim=-1)
    stats = {
        "mean_norm": float(norms.mean().item()),
        "mean_std": float(emb.std(dim=-1, unbiased=False).mean().item()),
        "mean_per_dim_std": float(emb.std(dim=0, unbiased=False).mean().item()),
        "effective_rank": _effective_rank(emb),
    }

    n = emb.shape[0]
    if n < 4:
        return stats

    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n, generator=generator)
    split = max(1, min(n - 1, int(0.8 * n)))
    train_idx, test_idx = perm[:split], perm[split:]
    x_train = emb[train_idx]
    x_test = emb[test_idx]

    per_target_r2 = {}
    if action is not None and action.shape[1] >= 2:
        per_target_r2["action_x"] = _ridge_r2(
            x_train, action[train_idx, 0], x_test, action[test_idx, 0], alpha=alpha
        )
        per_target_r2["action_y"] = _ridge_r2(
            x_train, action[train_idx, 1], x_test, action[test_idx, 1], alpha=alpha
        )
    if state is not None and state.shape[1] >= 7:
        per_target_r2["block_x"] = _ridge_r2(
            x_train, state[train_idx, 2], x_test, state[test_idx, 2], alpha=alpha
        )
        per_target_r2["block_y"] = _ridge_r2(
            x_train, state[train_idx, 3], x_test, state[test_idx, 3], alpha=alpha
        )
        per_target_r2["block_angle"] = _ridge_r2(
            x_train, state[train_idx, 4], x_test, state[test_idx, 4], alpha=alpha
        )
        per_target_r2["agent_vx"] = _ridge_r2(
            x_train, state[train_idx, 5], x_test, state[test_idx, 5], alpha=alpha
        )
        per_target_r2["agent_vy"] = _ridge_r2(
            x_train, state[train_idx, 6], x_test, state[test_idx, 6], alpha=alpha
        )

    stats["linear_r2_action_coord"] = _mean_metric(per_target_r2, ["action_x", "action_y"])
    stats["linear_r2_block_coord"] = _mean_metric(per_target_r2, ["block_x", "block_y"])
    stats["linear_r2_block_angle"] = _mean_metric(per_target_r2, ["block_angle"])
    stats["linear_r2_agent_velocity"] = _mean_metric(
        per_target_r2, ["agent_vx", "agent_vy"]
    )
    return stats


class ValidationEmbeddingStatsCallback(Callback):
    """Log lightweight embedding diagnostics on the validation subset."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self._emb_batches = []
        self._state_batches = []
        self._action_batches = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._emb_batches = []
        self._state_batches = []
        self._action_batches = []

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if trainer.sanity_checking or not isinstance(outputs, dict) or "emb" not in outputs:
            return

        emb = outputs["emb"].detach().float().cpu()
        self._emb_batches.append(_flatten_time_pairs(emb))

        if "state" in batch and torch.is_tensor(batch["state"]):
            state = batch["state"].detach().float().cpu()
            self._state_batches.append(_flatten_time_pairs(state))

        if "action" in batch and torch.is_tensor(batch["action"]):
            action = torch.nan_to_num(batch["action"].detach().float().cpu(), 0.0)
            self._action_batches.append(_flatten_time_pairs(action))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._emb_batches:
            return

        emb = torch.cat(self._emb_batches, dim=0)
        state = torch.cat(self._state_batches, dim=0) if self._state_batches else None
        action = torch.cat(self._action_batches, dim=0) if self._action_batches else None
        stats = compute_embedding_validation_stats(
            emb=emb,
            state=state,
            action=action,
            alpha=self.alpha,
        )
        metrics = {f"validate/{name}": value for name, value in stats.items()}
        for logger in getattr(trainer, "loggers", []):
            logger.log_metrics(metrics, step=trainer.global_step)
            logger.save()


def _flatten_subset_indices(dataset) -> tuple[torch.utils.data.Dataset, list[int]]:
    selected = list(range(len(dataset)))
    base_dataset = dataset
    while isinstance(base_dataset, torch.utils.data.Subset):
        selected = [int(base_dataset.indices[i]) for i in selected]
        base_dataset = base_dataset.dataset
    return base_dataset, selected


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def save_embeddings(model, dataset, path, batch_size: int, max_items: int | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    base_dataset, dataset_indices = _flatten_subset_indices(dataset)
    if max_items is not None:
        dataset_indices = dataset_indices[: max(0, int(max_items))]
    if not dataset_indices:
        raise ValueError("Dataset is empty; cannot save embeddings.")

    dataset = torch.utils.data.Subset(base_dataset, dataset_indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    emb_batches = []
    tensor_batches = {}

    with torch.inference_mode():
        for batch in loader:
            keys = [
                key
                for key, value in batch.items()
                if torch.is_tensor(value) and key != "pixels"
            ]
            batch = _move_batch_to_device(batch, device)
            if "action" in batch:
                batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            output = model.encode(batch)
            emb_batches.append(output["emb"].detach().float().cpu())
            for key in keys:
                tensor_batches.setdefault(key, []).append(batch[key].detach().float().cpu())

    if was_training:
        model.train()

    payload = {
        "dataset_indices": torch.tensor(dataset_indices, dtype=torch.long),
        "emb": torch.cat(emb_batches, dim=0),
    }
    for key, values in tensor_batches.items():
        payload[key] = torch.cat(values, dim=0)
    torch.save(payload, path)
