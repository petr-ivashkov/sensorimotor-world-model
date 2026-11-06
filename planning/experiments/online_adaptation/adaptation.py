"""Small, persistent online update loop for the existing SMWM objective.

Construct the model with ``eval.load_jepa_from_run`` (pretrained weights only)
or ``eval.build_jepa`` (scratch), then keep one learner across collection rounds.
An online checkpoint restores this learner's optimizer, never the source run's
optimizer. Input batches must already use the fixed source action normalizer.

Checkpoints preserve global RNG state. An explicit DataLoader generator and a
partially consumed iterator belong to the caller; checkpoint at round boundaries
and save any explicit generator separately for reproducible data ordering.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch


def _get(config: Any, *keys: str, default: Any = None) -> Any:
    value = config
    for key in keys:
        if isinstance(value, Mapping):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Incomplete online checkpoint RNG state")
    cuda_states = state["torch_cuda"]
    if cuda_states and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("Online checkpoint CUDA RNG device count does not match this host")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


class OnlineLearner:
    """Fine-tune all SMWM modules with AdamW and a fixed learning rate.

    ``precision='bf16'`` enables CUDA autocast. CPU always uses float32 so the
    same loop supports inexpensive smoke tests. No source optimizer, epoch
    counter, learning-rate schedule, or Lightning training state is reused.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        model: torch.nn.Module,
        source_cfg: Any,
        *,
        device: str = "cuda",
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-3,
        precision: str = "bf16",
        gradient_clip: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive and finite")
        if precision not in {"bf16", "float32"}:
            raise ValueError("precision must be 'bf16' or 'float32'")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("OnlineLearner supports CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use device='cpu' for a smoke test")
        self.precision = precision if self.device.type == "cuda" else "float32"
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bf16; use precision='float32'")
        self.history_size = int(_get(source_cfg, "wm", "history_size", default=1))
        if self.history_size < 1:
            raise ValueError("Source wm.history_size must be at least 1")
        self.inverse_weight = float(_get(source_cfg, "loss", "inverse", "weight", default=0.0))
        self.sigreg_weight = float(_get(source_cfg, "loss", "sigreg", "weight", default=0.0))
        if any(not math.isfinite(w) or w < 0 for w in (self.inverse_weight, self.sigreg_weight)):
            raise ValueError("Source loss weights must be nonnegative and finite")
        self.sigreg_kwargs = dict(_get(source_cfg, "loss", "sigreg", "kwargs", default={}))
        self.sigreg = None
        if self.sigreg_weight:
            # Import the repository implementation only when the source uses it.
            # The ordinary update path depends only on NumPy and PyTorch.
            from planning.module import SIGReg

            self.sigreg = SIGReg(**self.sigreg_kwargs).to(self.device)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # load_jepa_from_run freezes the model for evaluation. Online adaptation
        # explicitly trains every module, including the inverse model.
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.requires_grad_(True)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.gradient_clip = float(gradient_clip)
        self.update_steps = 0
        self._has_batchnorm = any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            for module in self.model.modules()
        )

    def _settings(self) -> dict[str, Any]:
        return {
            "history_size": self.history_size,
            "inverse_weight": self.inverse_weight,
            "sigreg_weight": self.sigreg_weight,
            "sigreg_kwargs": self.sigreg_kwargs,
            "precision": self.precision,
            "gradient_clip": self.gradient_clip,
        }

    def _losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.model.encode(batch)
        embeddings = output["emb"]
        required = self.history_size + 1
        if embeddings.ndim != 3 or embeddings.shape[1] < required:
            raise ValueError(f"Need embeddings [B,T,D] with at least {required} time steps")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot update on a batch with zero samples")
        if self._has_batchnorm and embeddings.shape[0] * self.history_size < 2:
            raise ValueError(
                "BatchNorm needs batch_size * history_size >= 2 for the prediction projector; "
                "use batch_size >= 2 and drop_last=True when history_size=1"
            )
        predicted = self.model.predict(
            embeddings[:, : self.history_size], output["act_emb"][:, : self.history_size]
        )
        losses = {"forward_loss": (predicted - embeddings[:, 1:required]).square().mean()}
        total = losses["forward_loss"]
        if self.inverse_weight:
            predicted_action = self.model.predict_action(embeddings[:, :-1], embeddings[:, 1:])
            losses["inverse_loss"] = (predicted_action - batch["action"][:, :-1]).square().mean()
            total = total + self.inverse_weight * losses["inverse_loss"]
        if self.sigreg_weight:
            losses["sigreg_loss"] = self.sigreg(embeddings.transpose(0, 1))
            total = total + self.sigreg_weight * losses["sigreg_loss"]
        losses["loss"] = total
        return losses

    def update(self, loader: Any, steps: int) -> dict[str, float | int]:
        """Take exactly ``steps`` updates, cycling a finite loader as needed.

        Report means over optimizer steps. ``update_steps`` counts all updates
        since online initialization; ``steps`` counts updates in this call.
        A loader yielding no batches raises instead of silently doing no work.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if steps == 0:
            return {"steps": 0, "update_steps": self.update_steps}
        self.model.train()
        iterator = iter(loader)
        sums: dict[str, float] = {}
        for _ in range(steps):
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                try:
                    raw_batch = next(iterator)
                except StopIteration as error:
                    raise ValueError("Online update loader is empty (check batch_size/drop_last)") from error
            if not isinstance(raw_batch, Mapping) or "action" not in raw_batch:
                raise ValueError("Online update batches must be mappings containing 'action'")
            batch = {
                key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.precision == "bf16" else nullcontext()
            )
            with context:
                losses = self._losses(batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite online adaptation loss; optimizer step was skipped")
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.update_steps += 1
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            sums["gradient_norm"] = sums.get("gradient_norm", 0.0) + float(grad_norm)
        return {**{name: value / steps for name, value in sums.items()},
                "steps": steps, "update_steps": self.update_steps}

    def state_dict(self) -> dict[str, Any]:
        """Return an independent snapshot suitable for ``torch.save``.

        The ``state_dict`` model prefix also supports the existing planning
        loader when the caller saves the source ``config.yaml`` beside it.
        """
        return copy.deepcopy({
            "online_checkpoint_version": self.CHECKPOINT_VERSION,
            "state_dict": {f"model.{key}": value for key, value in self.model.state_dict().items()},
            "optimizer_state": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "settings": self._settings(),
            "sigreg_state": self.sigreg.state_dict() if self.sigreg is not None else None,
            "rng": _rng_state(),
        })

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Strictly restore an online checkpoint; source Lightning files fail.

        Construct with matching objective, precision, and gradient clipping.
        Optimizer hyperparameters and momentum restore from the checkpoint.
        """
        required = {"online_checkpoint_version", "state_dict", "optimizer_state",
                    "update_steps", "settings", "sigreg_state", "rng"}
        if set(state) != required:
            raise ValueError(f"Online checkpoint keys mismatch: expected {sorted(required)}")
        if state["online_checkpoint_version"] != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported online checkpoint version")
        if state["settings"] != self._settings():
            raise ValueError("Online checkpoint settings do not match this learner")
        if isinstance(state["update_steps"], bool) or not isinstance(state["update_steps"], int) or state["update_steps"] < 0:
            raise ValueError("Invalid online checkpoint update_steps")
        if any(not key.startswith("model.") for key in state["state_dict"]):
            raise ValueError("Online checkpoint contains unexpected non-model weights")
        weights = {key[len("model."):]: value for key, value in state["state_dict"].items()}
        self.model.load_state_dict(weights, strict=True)
        if self.sigreg is not None:
            if state["sigreg_state"] is None:
                raise ValueError("Online checkpoint is missing SIGReg state")
            self.sigreg.load_state_dict(state["sigreg_state"], strict=True)
        elif state["sigreg_state"] is not None:
            raise ValueError("Online checkpoint unexpectedly contains SIGReg state")
        # Optimizer.load_state_dict can reuse tensor storage from its input.
        # Keep the caller's checkpoint immutable as resumed training advances.
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer_state"]))
        self.update_steps = state["update_steps"]
        _restore_rng(state["rng"])
        self.model.train()
