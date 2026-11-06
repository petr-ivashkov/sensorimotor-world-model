"""Pinned stable-worldmodel 0.0.6 collection and planning adapters.

Imports of the simulator and planner are deferred so configuration and HDF5
checks remain usable without loading graphical libraries. Round files are
immutable; this module never appends to an existing experience file.
"""

from copy import deepcopy
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


class FixedActionProcessor:
    """Use saved source-training action coordinates in the pinned policy API."""

    def __init__(self, stats):
        self.mean = np.asarray(stats["mean"])
        self.scale = np.asarray(stats["scale"])
        if self.mean.shape != self.scale.shape or self.mean.shape != (2,):
            raise ValueError("Push-T action mean and scale must both have shape (2,)")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all():
            raise ValueError("Action statistics must be finite")
        if np.any(self.scale <= 0):
            raise ValueError("Action scales must be positive")

    def transform(self, values):
        values = np.asarray(values)
        # Training performs source-dtype arithmetic and only then casts to float.
        dtype = values.dtype if values.dtype.kind == "f" else np.dtype("float32")
        return ((values - self.mean.astype(dtype)) / self.scale.astype(dtype)).astype(
            np.float32
        )

    def inverse_transform(self, values):
        values = np.asarray(values)
        return values * self.scale.astype(values.dtype) + self.mean.astype(values.dtype)


def make_world(env_config, *, seed, num_envs=1):
    """Construct a World from flat environment kwargs without changing config."""
    import stable_worldmodel as swm

    cfg = deepcopy(dict(env_config))
    env_name = cfg.pop("env_name", "swm/PushT-v1")
    if env_name == "swm/PushT-U-v0":
        from planning.experiments.online_adaptation.shape_env import register_shape_environments

        register_shape_environments()
    image_size = int(cfg.pop("img_size", 224))
    image_shape = tuple(cfg.pop("image_shape", (image_size, image_size)))
    cfg.pop("reset_options", None)
    cfg.pop("options", None)
    cfg.pop("seed", None)
    cfg.pop("num_envs", None)
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError("image_shape must contain two positive dimensions")
    return swm.World(
        env_name=env_name,
        num_envs=int(num_envs),
        image_shape=image_shape,
        seed=int(seed),
        history_size=int(cfg.pop("history_size", 1)),
        frame_skip=int(cfg.pop("frame_skip", 1)),
        max_episode_steps=int(cfg.pop("max_episode_steps", 200)),
        **cfg,
    )


def _reset_options(env_config):
    return deepcopy(env_config.get("reset_options", env_config.get("options")))


def _make_cem_policy(model, process, plan_config, solver_config, *, seed):
    import stable_worldmodel as swm
    from torchvision.transforms import v2

    if model is None or process is None:
        raise ValueError("CEM requires a model and saved source action processor")
    solver_cfg = dict(solver_config or {})
    if "device" not in solver_cfg:
        solver_cfg["device"] = str(next(model.parameters()).device)
    # Pinned CEM samples an unbounded Gaussian. Score the bounded action that
    # actually executes, including each raw action inside a five-step token.
    bounded_model = _BoundedCostModel(model, process["action"])
    solver = swm.solver.CEMSolver(model=bounded_model, seed=int(seed), **solver_cfg)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(**dict(plan_config or {})),
        process=process,
        # The model handles resize and ImageNet normalization itself.
        transform={"pixels": v2.ToImage(), "goal": v2.ToImage()},
    )


class _BoundedCostModel:
    """Clamp normalized candidate actions consistently with raw Push-T bounds."""

    def __init__(self, model, processor):
        self.model = model
        self.low = processor.transform(np.array([[-1., -1.]], dtype=np.float32))[0]
        self.high = processor.transform(np.array([[1., 1.]], dtype=np.float32))[0]

    def get_cost(self, infos, action_candidates):
        import torch

        raw_dim = len(self.low)
        if action_candidates.shape[-1] % raw_dim:
            raise ValueError("Action token does not contain complete raw Push-T actions")
        repeats = action_candidates.shape[-1] // raw_dim
        low = torch.as_tensor(self.low, device=action_candidates.device,
                              dtype=action_candidates.dtype).repeat(repeats)
        high = torch.as_tensor(self.high, device=action_candidates.device,
                               dtype=action_candidates.dtype).repeat(repeats)
        bounded = action_candidates.clamp(min=low, max=high)
        return self.model.get_cost(infos, bounded)


class _BoundedPolicy:
    """Clip executed raw actions; optionally rebuild CEM at each episode reset."""

    def __init__(self, factory, *, reset_each_episode=False):
        self.factory = factory
        self.reset_each_episode = reset_each_episode
        self.inner = None
        self.env = None

    def set_env(self, env):
        self.env = env
        self.inner = self.factory()
        self.inner.set_env(env)
        if getattr(self.inner, "seed", None) is not None:
            self.inner.set_seed(self.inner.seed)

    def get_action(self, infos):
        step = np.asarray(infos.get("step_idx", [0])).reshape(-1)
        if self.reset_each_episode and np.all(step == 0):
            self.inner = self.factory()
            self.inner.set_env(self.env)
        action = np.asarray(self.inner.get_action(deepcopy(infos)))
        if not np.isfinite(action).all():
            raise ValueError("Collection/planning policy produced a non-finite action")
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


def _collector_policy(policy_config, *, seed, model, process, plan_config, solver_config):
    cfg = dict(policy_config or {})
    kind = cfg.pop("policy", cfg.pop("type", cfg.pop("kind", "weak")))
    cfg.pop("seed", None)
    if kind == "weak":
        from stable_worldmodel.envs.pusht import WeakPolicy

        return _BoundedPolicy(lambda: WeakPolicy(seed=int(seed), **cfg)), kind
    if kind == "random":
        from stable_worldmodel.policy import RandomPolicy

        cfg.pop("dist_constraint", None)  # Common config; active only for WeakPolicy.
        if cfg:
            raise ValueError(f"Unexpected random collector options: {sorted(cfg)}")
        return _BoundedPolicy(lambda: RandomPolicy(seed=int(seed))), kind
    if kind == "cem":
        cfg.pop("dist_constraint", None)
        if cfg:
            raise ValueError(f"Unexpected CEM collector options: {sorted(cfg)}")
        history = int((plan_config or {}).get("history_len", 1))
        model_history = int(getattr(getattr(model, "predictor", None), "num_frames", 1))
        if history != 1 or model_history != 1:
            raise ValueError("Pinned HDF5 recording supports CEM collection only for H=1")
        return _BoundedPolicy(
            lambda: _make_cem_policy(model, process, plan_config, solver_config, seed=seed),
            reset_each_episode=True,
        ), kind
    raise ValueError(f"Unsupported collection policy: {kind!r}")


def _angle_distance(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def summarize_experience(path):
    """Count actual outgoing transitions and contact/motion without episode joins."""
    # Register Blosc filters when available; plain fixtures need no plugin.
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    counts = {"frames": 0, "raw_transitions": 0, "contact_transitions": 0}
    displacement, angular_change = [], []
    with h5py.File(path, "r") as f:
        lengths = np.asarray(f["ep_len"])
        offsets = np.asarray(f["ep_offset"])
        if len(lengths) == 0:
            raise ValueError("Collector produced no complete episodes")
        for offset, length in zip(offsets, lengths):
            offset, length = int(offset), int(length)
            actions = np.asarray(f["action"][offset : offset + length])
            states = np.asarray(f["state"][offset : offset + length])
            valid = np.isfinite(actions[:-1]).all(axis=-1)
            if not valid.all() or not np.isnan(actions[-1]).all():
                raise ValueError("Recorded actions violate outgoing-action/terminal-NaN alignment")
            counts["frames"] += length
            counts["raw_transitions"] += int(valid.sum())
            displacement.extend(np.linalg.norm(np.diff(states[:, 2:4], axis=0), axis=-1))
            angular_change.extend(_angle_distance(states[1:, 4], states[:-1, 4]))
            if "n_contacts" in f:
                contacts = np.asarray(f["n_contacts"][offset + 1 : offset + length])
                counts["contact_transitions"] += int(np.count_nonzero(contacts))
        counts["episodes"] = int(len(lengths))
        counts["episode_lengths"] = lengths.astype(int).tolist()
    transitions = counts["raw_transitions"]
    counts["contact_fraction"] = counts["contact_transitions"] / max(transitions, 1)
    counts["block_motion_mean"] = float(np.mean(displacement)) if displacement else 0.0
    counts["block_angle_change_mean"] = float(np.mean(angular_change)) if angular_change else 0.0
    counts["path"] = str(Path(path).resolve())
    return counts


def collect_round(path, env_config, *, seed, episodes, policy_config, model=None,
                  process=None, plan_config=None, solver_config=None):
    """Record a fresh immutable raw-step shard, closing the world on failure."""
    path = Path(path)
    if path.suffix != ".h5":
        raise ValueError("Experience paths must have the .h5 suffix")
    if path.exists():
        raise FileExistsError(f"Refusing to append to existing experience: {path}")
    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    cfg = deepcopy(dict(env_config))
    cfg.update(history_size=1, frame_skip=1)
    policy, kind = _collector_policy(
        policy_config, seed=seed, model=model, process=process,
        plan_config=plan_config, solver_config=solver_config,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a completed shard. A killed collection cannot masquerade as a
    # finished round, and link() refuses a concurrent writer's destination.
    with tempfile.TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as staging:
        staged_path = Path(staging) / path.name
        world = make_world(cfg, seed=seed, num_envs=1)
        was_training = getattr(model, "training", None)
        try:
            if model is not None:
                model.eval()
            world.set_policy(policy)
            if model is not None:
                import torch

                inference = torch.inference_mode()
            else:
                inference = nullcontext()
            with inference:
                world.record_dataset(
                    dataset_name=path.stem, episodes=int(episodes), seed=int(seed),
                    cache_dir=str(staged_path.parent), options=_reset_options(cfg),
                )
        finally:
            try:
                world.close()
            finally:
                if was_training is not None:
                    model.train(was_training)
        stats = summarize_experience(staged_path)
        if stats["episodes"] != int(episodes):
            raise ValueError("Recorded episode count does not match requested round size")
        os.link(staged_path, path)
    stats["path"] = str(path.resolve())
    stats.update(seed=int(seed), policy=kind, goal_source="environment_reset" if kind == "cem" else None)
    return stats


def _step_with_action(world, actions):
    (world.states, world.rewards, world.terminateds,
     world.truncateds, world.infos) = world.envs.step(actions)


def _last_state(world):
    return np.asarray(world.infos["state"])[0].reshape(-1, 7)[-1].copy()


def calibrate_shift(source_env_config, target_env_config, *, seed, steps=100,
                    policy_config=None):
    """Replay baseline-generated actions in both worlds; report, never select, a shift."""
    if int(steps) < 1:
        raise ValueError("Calibration steps must be positive")
    source_cfg, target_cfg = deepcopy(source_env_config), deepcopy(target_env_config)
    for cfg in (source_cfg, target_cfg):
        cfg.update(history_size=1, frame_skip=1)
        cfg["max_episode_steps"] = max(int(cfg.get("max_episode_steps", 200)), int(steps) + 1)
    source = make_world(source_cfg, seed=seed)
    target = None
    try:
        target = make_world(target_cfg, seed=seed)
        source.reset(seed=int(seed), options=_reset_options(source_cfg))
        target.reset(seed=int(seed), options=_reset_options(target_cfg))
        initial_difference = float(np.linalg.norm(_last_state(source) - _last_state(target)))
        policy, _ = _collector_policy(
            policy_config or {"type": "weak"}, seed=seed,
            model=None, process=None, plan_config=None, solver_config=None,
        )
        source.set_policy(policy)
        position_errors, angle_errors, contact_counts = [], [], [0, 0]
        for _ in range(int(steps)):
            actions = policy.get_action(source.infos)
            _step_with_action(source, actions)
            _step_with_action(target, actions.copy())
            src_state, tgt_state = _last_state(source), _last_state(target)
            position_errors.append(float(np.linalg.norm(src_state[2:4] - tgt_state[2:4])))
            angle_errors.append(float(_angle_distance(src_state[4], tgt_state[4])))
            for i, world in enumerate((source, target)):
                contact_counts[i] += int(np.any(np.asarray(world.infos.get("n_contacts", 0)) > 0))
            if any(np.any(w.terminateds) or np.any(w.truncateds) for w in (source, target)):
                break
        actual_steps = len(position_errors)
        return {
            "seed": int(seed), "steps": actual_steps,
            "initial_state_difference": initial_difference,
            "mean_block_position_divergence": float(np.mean(position_errors)),
            "final_block_position_divergence": position_errors[-1],
            "mean_block_angle_divergence": float(np.mean(angle_errors)),
            "final_block_angle_divergence": angle_errors[-1],
            "source_contact_fraction": contact_counts[0] / actual_steps,
            "target_contact_fraction": contact_counts[1] / actual_steps,
            "measurable_shift": bool(max(position_errors) > 1e-6 or max(angle_errors) > 1e-6),
            "has_contacts": bool(max(contact_counts) > 0),
        }
    finally:
        source.close()
        if target is not None:
            target.close()


def _validate_tasks(dataset, tasks):
    episodes = np.asarray(tasks["episode_ids"], dtype=int)
    starts = np.asarray(tasks["start_steps"], dtype=int)
    offset = int(tasks["goal_offset_steps"])
    if episodes.ndim != 1 or starts.shape != episodes.shape or len(episodes) == 0:
        raise ValueError("Tasks require equal nonempty episode_ids/start_steps lists")
    if offset < 2 or np.any(starts < 0) or np.any(episodes < 0) or np.any(episodes >= len(dataset.lengths)):
        raise ValueError("Invalid frozen planning task index or goal offset")
    if np.any(starts + offset > np.asarray(dataset.lengths)[episodes]):
        raise ValueError("A frozen planning goal extends beyond its episode")
    return episodes, starts, offset


def evaluate_planning(model, dataset_path, env_config, action_stats, tasks, *, seed,
                      plan_config, solver_config, eval_budget, device):
    """Evaluate fixed held-out future-goal tasks with the repository's CEM API."""
    import stable_worldmodel as swm

    path = Path(dataset_path)
    dataset = swm.data.HDF5Dataset(
        path.stem, cache_dir=str(path.parent),
        keys_to_cache=["action", "state", "proprio"],
    )
    world = None
    was_training = model.training
    try:
        episodes, starts, offset = _validate_tasks(dataset, tasks)
        cfg = deepcopy(dict(env_config))
        history = int(plan_config.get("history_len", 1))
        cfg["history_size"] = (history - 1) * int(plan_config.get("action_block", 1)) + 1
        cfg["frame_skip"] = 1
        cfg["max_episode_steps"] = max(int(cfg.get("max_episode_steps", 200)), int(eval_budget) + 1, offset)
        world = make_world(cfg, seed=seed, num_envs=len(episodes))
        model.eval()
        solver_cfg = dict(solver_config)
        solver_cfg["device"] = device
        policy = _BoundedPolicy(lambda: _make_cem_policy(
            model, {"action": FixedActionProcessor(action_stats)}, plan_config,
            solver_cfg, seed=seed,
        ))
        world.set_policy(policy)
        metrics = world.evaluate_from_dataset(
            dataset, episodes_idx=episodes.tolist(), start_steps=starts.tolist(),
            goal_offset_steps=offset, eval_budget=int(eval_budget),
            callables=[
                {"method": "_set_state", "args": {"state": {"value": "state"}}},
                {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
            ], save_video=False,
        )
        # Pinned load_chunk uses exclusive end: preserve its goal-offset convention.
        goals = np.stack([
            dataset._load_slice(int(ep), int(start + offset - 1), int(start + offset))["state"][0].numpy()
            for ep, start in zip(episodes, starts)
        ])
        final = np.asarray(world.infos["state"])[:, -1]
        metrics.update(
            final_block_position_error=float(np.linalg.norm(final[:, 2:4] - goals[:, 2:4], axis=-1).mean()),
            final_block_angle_error=float(_angle_distance(final[:, 4], goals[:, 4]).mean()),
            final_agent_position_error=float(np.linalg.norm(final[:, :2] - goals[:, :2], axis=-1).mean()),
            num_tasks=len(episodes), actual_goal_offset_steps=offset - 1,
            action_clipping=True, action_bounds_in_cost=True,
        )
        return _json_values(metrics)
    finally:
        if world is not None:
            world.close()
        if getattr(dataset, "h5_file", None) is not None:
            dataset.h5_file.close()
        model.train(was_training)


def _json_values(value):
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
