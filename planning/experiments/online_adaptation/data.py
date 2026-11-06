"""Small, immutable HDF5 experience shards using stable-worldmodel's schema.

Read only sampled pixels, keep action/state metadata in RAM, and open one HDF5
handle per DataLoader worker. Source action coordinates stay fixed across rounds.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import hdf5plugin  # noqa: F401 -- registers source dataset compression filters
except ImportError:
    pass


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_statistics(path: Path) -> dict:
    """Match training's raw-action mean and sample standard deviation (ddof=1)."""
    with h5py.File(path, "r") as handle:
        actions = handle["action"][:]
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("Initial Push-T pilot requires raw two-dimensional actions")
    actions = actions[np.isfinite(actions).all(axis=1)]
    if len(actions) < 2:
        raise ValueError("Source needs at least two finite action rows")
    tensor = torch.from_numpy(actions)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    mean, scale = tensor.mean(0), tensor.std(0, correction=1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Source action normalization has zero or invalid scale")
    return {"mean": mean.tolist(), "scale": scale.tolist(), "count": len(actions),
            "ddof": 1, "source": str(Path(path).resolve()), "dtype": str(actions.dtype)}


def episode_metadata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
        rows = len(handle["pixels"])
        for key in ("action", "state"):
            if len(handle[key]) != rows:
                raise ValueError(f"{path}: {key} and pixels have different row counts")
    if (lengths.ndim != 1 or lengths.shape != offsets.shape or len(lengths) == 0
            or (lengths < 1).any()):
        raise ValueError(f"{path}: invalid episode metadata")
    expected = np.concatenate(([0], np.cumsum(lengths[:-1])))
    if not np.array_equal(offsets, expected) or int(lengths.sum()) != rows:
        raise ValueError(f"{path}: episodes must be contiguous and span all rows")
    return lengths, offsets


def split_episodes(path: Path, *, seed: int, fit_fraction: float = 0.5) -> dict:
    lengths, _ = episode_metadata(path)
    if len(lengths) < 2 or not 0 < fit_fraction < 1:
        raise ValueError("At least two episodes and a fraction in (0,1) are required")
    order = np.random.default_rng(seed).permutation(len(lengths))
    count = max(1, min(len(lengths) - 1, int(len(lengths) * fit_fraction)))
    return {"fit": sorted(order[:count].tolist()), "test": sorted(order[count:].tolist())}


def fixed_tasks(path: Path, *, episodes: list[int], count: int,
                goal_offset_steps: int, seed: int, min_block_displacement: float = 0.0) -> dict:
    """One fixed start/goal pair per episode, following pinned evaluator semantics."""
    lengths, _ = episode_metadata(path)
    if count < 1 or goal_offset_steps < 2:
        raise ValueError("Positive task count and goal_offset_steps >= 2 required")
    if (len(set(episodes)) != len(episodes)
            or any(ep < 0 or ep >= len(lengths) for ep in episodes)):
        raise ValueError("Task episode IDs must be unique and inside the shard")
    if not np.isfinite(min_block_displacement) or min_block_displacement < 0:
        raise ValueError("min_block_displacement must be finite and nonnegative")
    _, offsets = episode_metadata(path)
    valid_starts = {}
    with h5py.File(path, "r") as handle:
        for ep in episodes:
            starts = np.arange(0, lengths[ep] - goal_offset_steps + 1, dtype=np.int64)
            if min_block_displacement and len(starts):
                offset = int(offsets[ep])
                xy = handle["state"][offset:offset+int(lengths[ep]), 2:4]
                movement = np.linalg.norm(xy[starts+goal_offset_steps-1] - xy[starts], axis=1)
                starts = starts[movement >= min_block_displacement]
            if len(starts):
                valid_starts[int(ep)] = starts
    candidates = list(valid_starts)
    if len(candidates) < count:
        raise ValueError(f"Need {count} held-out episodes satisfying length/motion requirements; found {len(candidates)}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=count, replace=False).tolist()
    return {"episode_ids": chosen,
            "start_steps": [int(rng.choice(valid_starts[ep])) for ep in chosen],
            "goal_offset_steps": goal_offset_steps}


class ExperienceDataset(Dataset):
    """Episode-safe clips; terminal/padded actions never supervise a transition.

    Like SWM's HDF5Dataset, stride observations by K and concatenate K consecutive
    raw actions into each token. The final action token is zero padding and is
    unused by the objective/rollout. This avoids consuming terminal NaN actions.
    """

    def __init__(self, path: Path, statistics: dict, *, frameskip: int = 5,
                 num_steps: int = 2, episodes: list[int] | None = None,
                 max_clips: int | None = None, sample_seed: int = 0,
                 clip_stride: int = 1):
        self._handle = None
        self._pixels = None
        self._pid = None
        self.path = Path(path).resolve()
        self.frameskip, self.num_steps = int(frameskip), int(num_steps)
        if self.frameskip < 1 or self.num_steps < 2 or clip_stride < 1:
            raise ValueError("frameskip/clip_stride must be positive; num_steps >= 2")
        self.lengths, self.offsets = episode_metadata(self.path)
        self.episodes = list(range(len(self.lengths))) if episodes is None else list(episodes)
        if len(set(self.episodes)) != len(self.episodes):
            raise ValueError("Duplicate episode IDs")
        if any(ep < 0 or ep >= len(self.lengths) for ep in self.episodes):
            raise ValueError("Episode ID outside shard")
        with h5py.File(self.path, "r") as handle:
            self.actions = handle["action"][:]
            self.states = handle["state"][:]
        if self.actions.ndim != 2 or self.actions.shape[1] != len(statistics["mean"]):
            raise ValueError("Action shape differs from source normalization")
        dtype = self.actions.dtype if np.issubdtype(self.actions.dtype, np.floating) else np.float32
        self.mean = np.asarray(statistics["mean"], dtype=dtype)
        self.scale = np.asarray(statistics["scale"], dtype=dtype)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Invalid action statistics")
        # Any nonfinite executed action invalidates a clip; terminal NaNs are excluded.
        bad = ~np.isfinite(self.actions).all(axis=1)
        prefix = np.concatenate(([0], np.cumsum(bad)))
        self.span_actions = (self.num_steps - 1) * self.frameskip
        clips = []
        for ep in self.episodes:
            offset, length = int(self.offsets[ep]), int(self.lengths[ep])
            starts = np.arange(0, length - self.span_actions, clip_stride, dtype=np.int64)
            valid = (prefix[offset + starts + self.span_actions] - prefix[offset + starts]) == 0
            clips.extend((ep, int(start)) for start in starts[valid])
        if max_clips is not None:
            if max_clips < 1:
                raise ValueError("max_clips must be positive")
            if len(clips) > max_clips:
                picks = np.sort(np.random.default_rng(sample_seed).choice(
                    len(clips), size=max_clips, replace=False))
                clips = [clips[int(i)] for i in picks]
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = h5py.File(self.path, "r", rdcc_nbytes=32 * 1024 * 1024)
            self._pixels = self._handle["pixels"]
            self._pid = os.getpid()
        ep, start = self.clips[index]
        begin = int(self.offsets[ep]) + start
        end = begin + self.span_actions + 1
        # Compressed SWM pixels are chunked in 100-frame groups. A strided
        # HDF5 hyperslab has expensive scatter-copy overhead even with cached
        # chunks. Scalar frame reads are contiguous and reuse the dataset's
        # raw chunk cache. Read exactly the same selected frames, in order.
        pixels = np.stack([self._pixels[row] for row in range(begin, end, self.frameskip)])
        if pixels.ndim != 4 or pixels.shape[-1] not in (1, 3):
            raise ValueError("Expected SWM pixels in (time,height,width,channels) layout")
        raw = self.actions[begin:begin + self.span_actions]
        normalized = ((raw - self.mean) / self.scale).astype(np.float32)
        action = np.zeros((self.num_steps, self.frameskip * len(self.mean)), dtype=np.float32)
        action[:-1] = normalized.reshape(self.num_steps - 1, -1)
        return {"pixels": torch.from_numpy(pixels).permute(0, 3, 1, 2),
                "action": torch.from_numpy(action),
                "state": torch.from_numpy(self.states[begin:end:self.frameskip].copy()).float(),
                "ep_idx": torch.tensor(ep, dtype=torch.int64),
                "step_idx": torch.tensor(start, dtype=torch.int64)}

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None
        self._pixels = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pixels"] = None
        state["_pid"] = None
        return state

    def __del__(self):
        self.close()
