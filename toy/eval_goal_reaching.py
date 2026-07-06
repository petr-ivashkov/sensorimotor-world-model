"""Oracle-subgoal goal-reaching evaluation for trained (multistep) world models.

Protocol — deliberately an oracle over subgoals, so failures are attributable
to *decoding* rather than subgoal generation:

  1. Roll a ground-truth episode s_0 … s_T with the data-generating dynamics
     (a k=T trajectory segment, so every frame is in-distribution).
  2. Keep every ``spacing``-th state (plus s_T) as subgoals.
  3. Receding-horizon replanning: at every env step re-encode the current
     observation and decode the action toward the active subgoal pairwise with
     the inverse model, â = h_ψ(f_θ(o_cur), f_θ(o_subgoal)); execute â.
     A subgoal is advanced when reached (all controllable DOFs within
     threshold) or when its step budget is exhausted.
  4. Score against the final goal s_T on the *controllable* DOFs only;
     uncontrollable (RANDOM) dots keep moving during execution and their
     residual error is reported separately (it should stay at chance).

Since the inverse head is trained to predict the *mean* per-step action over
its horizon k, executing it closed-loop is a proportional controller: each
step covers ≈ 1/k of the remaining gap, so the per-subgoal budget scales with
max(spacing, horizon_k); ``horizon_k`` is read from the run's config.yaml.

Usage (mirrors train_decoder.py: the run's saved config.yaml is the single
source of truth for the world and the models):

    python eval_goal_reaching.py --config experiments/multistep_inverse/config_eval_single_dot.yaml

Writes ``goal_reaching.pt`` into each run directory and prints a summary table.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.structured_dot_world import (
    MotionType,
    StructuredDotWorldSegmentDataset,
    render_dots,
    step_positions,
)
from datasets.sprite_world import (
    SpriteWorldSegmentDataset,
    render_sprite,
    step_pose,
)
from models import CNNEncoder, InverseModel
from train import build_world, load_config


# ─────────────────────────────────────────────────────────────────
#  World adapters
# ─────────────────────────────────────────────────────────────────

class StructuredWorldAdapter:
    """Structured dot world: state = (num_dots, 2) positions."""

    def __init__(self, world_cfg, path_steps: int):
        self.cfg = world_cfg
        self._ds = StructuredDotWorldSegmentDataset(
            world_cfg, num_samples=0, seed=0, num_steps=path_steps)
        self.ctrl_dots, self.unctrl_dots = [], []
        for group, (start, end) in zip(world_cfg.groups, world_cfg.group_ranges()):
            if group.motion_type in (MotionType.INDEPENDENT, MotionType.COUPLED):
                self.ctrl_dots.extend(range(start, end))
            elif group.motion_type is MotionType.RANDOM:
                self.unctrl_dots.extend(range(start, end))
            # STATIC dots never move and trivially stay at goal: count neither.

    def sample_path(self, rng):
        states, _ = self._ds.sample_segment(rng)
        return states

    def render(self, state) -> torch.Tensor:
        return torch.from_numpy(render_dots(state, self._ds._color_indices, self.cfg))

    def step(self, state, action, rng):
        return step_positions(self.cfg, state, action, rng)

    def metrics(self, state, goal) -> dict:
        d = np.linalg.norm(np.asarray(state, float) - np.asarray(goal, float), axis=-1)
        ctrl = d[self.ctrl_dots]
        return {
            "pos_err": float(ctrl.mean()),
            "pos_err_max": float(ctrl.max()),
            "theta_err": float("nan"),
            "unctrl_err": float(d[self.unctrl_dots].mean()) if self.unctrl_dots else float("nan"),
        }


class SpriteWorldAdapter:
    """Sprite world: state = (3,) pose (x, y, θ)."""

    def __init__(self, world_cfg, path_steps: int):
        self.cfg = world_cfg
        self._ds = SpriteWorldSegmentDataset(
            world_cfg, num_samples=0, seed=0, num_steps=path_steps)
        mask = world_cfg.control_mask
        self.ctrl_xy = [i for i in (0, 1) if mask[i]]
        self.unctrl_xy = [i for i in (0, 1) if not mask[i]]
        self.theta_controlled = bool(mask[2])

    def sample_path(self, rng):
        states, _ = self._ds.sample_segment(rng)
        return states

    def render(self, state) -> torch.Tensor:
        return torch.from_numpy(render_sprite(state, self.cfg))

    def step(self, state, action, rng):
        return step_pose(self.cfg, state, action, rng)

    def metrics(self, state, goal) -> dict:
        state, goal = np.asarray(state, float), np.asarray(goal, float)
        pos_err = (float(np.linalg.norm(state[self.ctrl_xy] - goal[self.ctrl_xy]))
                   if self.ctrl_xy else float("nan"))
        dtheta = (state[2] - goal[2] + np.pi) % (2.0 * np.pi) - np.pi
        return {
            "pos_err": pos_err,
            "pos_err_max": pos_err,
            "theta_err": abs(float(dtheta)) if self.theta_controlled else float("nan"),
            "unctrl_err": (float(np.linalg.norm(state[self.unctrl_xy] - goal[self.unctrl_xy]))
                           if self.unctrl_xy else float("nan")),
        }


def reached(metrics: dict, thresholds: dict) -> bool:
    """All controllable DOFs within threshold (NaN entries don't apply)."""
    pos_ok = math.isnan(metrics["pos_err_max"]) or metrics["pos_err_max"] <= thresholds["pos"]
    theta_ok = math.isnan(metrics["theta_err"]) or metrics["theta_err"] <= thresholds["theta"]
    return pos_ok and theta_ok


# ─────────────────────────────────────────────────────────────────
#  Episode rollout
# ─────────────────────────────────────────────────────────────────

def run_episode(adapter, encoder, inv_model, action_scale, states, spacing,
                thresholds, budget, rng, device):
    """Follow oracle subgoals from s_0 toward s_T; return the episode record."""
    T = len(states) - 1
    subgoal_idx = list(range(spacing, T + 1, spacing))
    if not subgoal_idx or subgoal_idx[-1] != T:   # spacing > T degrades to goal-only
        subgoal_idx.append(T)

    cur = np.asarray(states[0], dtype=np.float64)
    trace = [cur.copy()]
    steps = 0
    for gi in subgoal_idx:
        goal = states[gi]
        z_g = encoder(adapter.render(goal)[None].to(device))
        for _ in range(budget):
            if reached(adapter.metrics(cur, goal), thresholds):
                break
            z = encoder(adapter.render(cur)[None].to(device))
            a_hat = (inv_model(z, z_g) * action_scale)[0].cpu().numpy().astype(np.float64)
            cur = adapter.step(cur, a_hat, rng)
            steps += 1
            trace.append(cur.copy())

    final = adapter.metrics(cur, states[-1])
    init = adapter.metrics(states[0], states[-1])
    return {
        "success": reached(final, thresholds),
        "steps": steps,
        "pos_err_init": init["pos_err"],
        "trace": np.stack(trace),
        **final,
    }


# ─────────────────────────────────────────────────────────────────
#  Aggregation / reporting
# ─────────────────────────────────────────────────────────────────

EPISODE_KEYS = ("pos_err", "pos_err_max", "theta_err", "unctrl_err", "pos_err_init", "steps")


def _nanmean(vals: np.ndarray) -> float:
    ok = vals[~np.isnan(vals)]
    return float(ok.mean()) if ok.size else float("nan")


def aggregate(records: list[dict]) -> dict:
    out = {"episodes": len(records),
           "success_rate": float(np.mean([r["success"] for r in records]))}
    for key in EPISODE_KEYS:
        vals = np.array([r[key] for r in records], dtype=float)
        out[f"{key}_mean"] = _nanmean(vals)
    pos = np.array([r["pos_err"] for r in records], dtype=float)
    out["pos_err_median"] = float(np.median(pos[~np.isnan(pos)])) if (~np.isnan(pos)).any() else float("nan")
    return out


def print_table(run_name: str, horizon_k: int, summary: dict) -> None:
    cols = ["spacing", "success", "pos_err", "pos_med", "theta_err", "unctrl_err", "init_err", "steps"]
    print(f"\n  {run_name}  (horizon_k={horizon_k})")
    print("  " + "  ".join(f"{c:>10}" for c in cols))
    for m, agg in summary.items():
        row = [f"{m:>10d}",
               f"{agg['success_rate']:>10.2f}",
               f"{agg['pos_err_mean']:>10.2f}",
               f"{agg['pos_err_median']:>10.2f}",
               f"{agg['theta_err_mean']:>10.3f}",
               f"{agg['unctrl_err_mean']:>10.2f}",
               f"{agg['pos_err_init_mean']:>10.2f}",
               f"{agg['steps_mean']:>10.1f}"]
        print("  " + "  ".join(row))


# ─────────────────────────────────────────────────────────────────
#  Per-run evaluation
# ─────────────────────────────────────────────────────────────────

def load_frozen_models(run_dir: Path, m_cfg: dict, image_size: int, action_dim: int, device):
    """Construct encoder f_θ + inverse h_ψ, load their weights, and freeze them."""
    latent_dim, hidden_dim = int(m_cfg["latent_dim"]), int(m_cfg["hidden_dim"])
    encoder = CNNEncoder(latent_dim=latent_dim, image_size=image_size).to(device)
    inv_model = InverseModel(latent_dim=latent_dim, action_dim=action_dim,
                             hidden_dim=hidden_dim).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    encoder.load_state_dict(state["encoder"])
    inv_model.load_state_dict(state["inverse"])
    for m in (encoder, inv_model):
        m.eval()
        m.requires_grad_(False)
    return encoder, inv_model


@torch.inference_mode()
def evaluate_run(run_dir: Path, ev: dict, device) -> dict:
    """Evaluate one trained run over all subgoal spacings; save goal_reaching.pt."""
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ds_cfg, m_cfg, t_cfg = run_cfg["dataset"], run_cfg["model"], run_cfg["training"]
    horizon_k = int(t_cfg.get("horizon_k", 1))

    world_cfg, _, action_scale = build_world(ds_cfg)
    if world_cfg.action_dim == 0:
        raise ValueError(f"{run_dir.name}: goal-reaching eval needs an actionable "
                         f"world (action_dim > 0).")
    if isinstance(action_scale, torch.Tensor):
        action_scale = action_scale.to(device)

    kind = str(ds_cfg.get("world", "structured")).lower()
    Adapter = StructuredWorldAdapter if kind == "structured" else SpriteWorldAdapter
    path_steps = int(ev["path_steps"])
    adapter = Adapter(world_cfg, path_steps)

    encoder, inv_model = load_frozen_models(
        run_dir, m_cfg, world_cfg.image_size, world_cfg.action_dim, device)

    spacings = [int(m) for m in ev["subgoal_spacings"]]
    episodes = int(ev["episodes"])
    seed = int(ev["seed"])
    thresholds = {"pos": float(ev["success_threshold"]),
                  "theta": float(ev["theta_threshold"])}

    records: dict[int, list[dict]] = {m: [] for m in spacings}
    for e in range(episodes):
        # One path per episode index, shared across spacings and runs (same
        # eval seed), so comparisons are paired.
        states = adapter.sample_path(np.random.default_rng([seed, e]))
        for m in spacings:
            budget = int(np.ceil(float(ev["budget_factor"]) * max(m, horizon_k))) \
                     + int(ev["budget_slack"])
            rec = run_episode(adapter, encoder, inv_model, action_scale, states,
                              m, thresholds, budget,
                              np.random.default_rng([seed, e, m]), device)
            records[m].append(rec)

    summary = {m: aggregate(records[m]) for m in spacings}
    detail = {m: {"success": np.array([r["success"] for r in records[m]]),
                  **{k: np.array([r[k] for r in records[m]], dtype=float)
                     for k in EPISODE_KEYS}}
              for m in spacings}
    if bool(ev.get("save_traces", True)):
        for m in spacings:
            detail[m]["traces"] = [r["trace"] for r in records[m]]

    out = {"eval": ev, "world": kind, "horizon_k": horizon_k,
           "spacings": spacings, "summary": summary, "episodes": detail}
    torch.save(out, run_dir / "goal_reaching.pt")
    print_table(run_dir.name, horizon_k, summary)
    print(f"  Saved goal_reaching.pt to {run_dir}")
    return out


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle-subgoal goal-reaching evaluation: decode ground-truth "
                    "subgoals pairwise with the inverse model under receding-horizon "
                    "replanning.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    ev = cfg["eval"]
    model_runs = cfg["model_runs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config       : {config_path}")
    print(f"Device       : {device}")
    print(f"Settings     : episodes={ev['episodes']} path_steps={ev['path_steps']} "
          f"spacings={ev['subgoal_spacings']} threshold={ev['success_threshold']}px")

    for run in model_runs:
        run_dir = (config_path.parent / "results" / run).resolve()
        if not run_dir.exists():
            print(f"\n  [missing] {run} ({run_dir})")
            continue
        evaluate_run(run_dir, ev, device)


if __name__ == "__main__":
    main()
