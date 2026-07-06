"""Aggregate multistep-inverse results into one table.

For every run under results/ this reports, per (env, horizon_k):

  - final eval forward / inverse losses            (train_history.pt)
  - effective rank of the final latent snapshot    (embeddings.pt; the
    structured_dots notebook's PCA convention) vs. the true controllable dim
  - held-out ridge-probe R² of z_t for controllable and uncontrollable state
    coordinates — "does multistep IDR still recover the controllable DOFs"
  - goal-reaching success rate per subgoal spacing (goal_reaching.pt)

Writes results/summary.csv and prints the table.

Usage:  python aggregate_results.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.structured_dot_world import MotionType          # noqa: E402
from train import build_world                                  # noqa: E402


# ─────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────

def effective_rank(Z: np.ndarray, eps: float = 1e-12) -> float:
    """exp(entropy of normalized PCA eigenvalues) — as in analyze.ipynb."""
    Zc = Z - Z.mean(axis=0)
    _, S, _ = np.linalg.svd(Zc, full_matrices=False)
    ev = np.maximum((S ** 2) / max(1, Z.shape[0] - 1), eps)
    e = ev[ev > eps]
    p = e / e.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def ridge_r2(Z: np.ndarray, y: np.ndarray, alpha: float = 1e-3,
             test_frac: float = 0.2) -> np.ndarray:
    """Held-out R² per target column of a ridge probe z_t → y."""
    n_test = max(1, int(len(Z) * test_frac))
    Ztr, Zte = Z[:-n_test], Z[-n_test:]
    ytr, yte = y[:-n_test], y[-n_test:]
    mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0) + 1e-8
    Xtr, Xte = (Ztr - mu) / sd, (Zte - mu) / sd
    ym = ytr.mean(axis=0)
    w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]),
                        Xtr.T @ (ytr - ym))
    pred = Xte @ w + ym
    ss_res = ((yte - pred) ** 2).sum(axis=0)
    ss_tot = ((yte - yte.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def probe_r2(run_cfg: dict, z_t: np.ndarray, positions: np.ndarray):
    """(R²_controllable, R²_uncontrollable) of a z_t → state probe.

    Structured world: per-dot (x, y) coordinates, grouped by motion type
    (STATIC excluded — zero variance).  Sprite world: x, y and θ (probed as
    sin θ, cos θ), grouped by the control mask.
    """
    ds_cfg = run_cfg["dataset"]
    world_cfg, _, _ = build_world(ds_cfg)
    kind = str(ds_cfg.get("world", "structured")).lower()

    if kind == "structured":
        ctrl_cols, unctrl_cols = [], []
        for group, (start, end) in zip(world_cfg.groups, world_cfg.group_ranges()):
            cols = [c for d in range(start, end) for c in (2 * d, 2 * d + 1)]
            if group.motion_type in (MotionType.INDEPENDENT, MotionType.COUPLED):
                ctrl_cols.extend(cols)
            elif group.motion_type is MotionType.RANDOM:
                unctrl_cols.extend(cols)
        r2 = ridge_r2(z_t, positions)
        r2_ctrl = float(r2[ctrl_cols].mean()) if ctrl_cols else float("nan")
        r2_unctrl = float(r2[unctrl_cols].mean()) if unctrl_cols else float("nan")
        return r2_ctrl, r2_unctrl

    # Sprite: positions holds the pose (x, y, θ).
    mask = world_cfg.control_mask
    theta = positions[:, 2]
    targets = np.column_stack([positions[:, 0], positions[:, 1],
                               np.sin(theta), np.cos(theta)])
    r2 = ridge_r2(z_t, targets)
    per_dof = [r2[0], r2[1], 0.5 * (r2[2] + r2[3])]      # x, y, θ
    ctrl = [per_dof[i] for i in range(3) if mask[i]]
    unctrl = [per_dof[i] for i in range(3) if not mask[i]]
    r2_ctrl = float(np.mean(ctrl)) if ctrl else float("nan")
    r2_unctrl = float(np.mean(unctrl)) if unctrl else float("nan")
    return r2_ctrl, r2_unctrl


# ─────────────────────────────────────────────────────────────────
#  Collection
# ─────────────────────────────────────────────────────────────────

def collect_run(run_dir: Path) -> dict | None:
    if not (run_dir / "config.yaml").exists():
        return None
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    world_cfg, _, _ = build_world(run_cfg["dataset"])
    row = {
        "run": run_dir.name,
        "world": str(run_cfg["dataset"].get("world", "structured")).lower(),
        "horizon_k": int(run_cfg["training"].get("horizon_k", 1)),
        "true_dim": world_cfg.action_dim,
    }

    hist_path = run_dir / "train_history.pt"
    if hist_path.exists():
        hist = torch.load(hist_path, map_location="cpu", weights_only=False)
        row["eval_fwd"] = hist["eval"]["fwd"][-1]
        row["eval_inv"] = hist["eval"]["inv"][-1]

    emb_path = run_dir / "embeddings.pt"
    if emb_path.exists():
        emb = torch.load(emb_path, map_location="cpu", weights_only=False)
        last = max(emb["snapshots"])
        snap = emb["snapshots"][last]
        Z = torch.cat([snap["z_t"], snap["z_tp1"]]).numpy()
        row["eff_rank"] = effective_rank(Z)
        r2_ctrl, r2_unctrl = probe_r2(
            run_cfg, snap["z_t"].numpy(), emb["positions"].numpy())
        row["r2_ctrl"], row["r2_unctrl"] = r2_ctrl, r2_unctrl

    gr_path = run_dir / "goal_reaching.pt"
    if gr_path.exists():
        gr = torch.load(gr_path, map_location="cpu", weights_only=False)
        for m in gr["spacings"]:
            row[f"succ@{m}"] = gr["summary"][m]["success_rate"]
            row[f"pos_err@{m}"] = gr["summary"][m]["pos_err_mean"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path,
                        default=EXPERIMENT_DIR / "results")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.results_dir.iterdir() if args.results_dir.exists() else []):
        if run_dir.is_dir():
            row = collect_run(run_dir)
            if row is not None:
                rows.append(row)
    if not rows:
        print(f"No runs found under {args.results_dir}. Run ./run.sh first.")
        return

    df = pd.DataFrame(rows).sort_values(["world", "run"]).reset_index(drop=True)
    out_csv = args.results_dir / "summary.csv"
    df.to_csv(out_csv, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
