"""Verification for the multistep (first-action) inverse regularizer + goal-reaching eval.

Run from toy/ (also works under pytest):

    python tests/test_multistep_inverse.py

Covers, in order:
  1. segment datasets with k_min=k_max=1 are bitwise identical to the
     transition datasets (the k=1 single-step IDR path is definitional)
  2. random-k segments are self-consistent (k in range, deterministic per
     index, first action matches the first transition, states in-canvas)
  3. MultistepInverseModel is genuinely conditioned on k
  4. train_multistep.compute_losses at k_max=1 == train.compute_losses exactly
  5. a short train_multistep.py k_max=1 run == train.py bitwise (both worlds)
  6. closed-loop steppers replay recorded segment actions exactly
  7. eval_goal_reaching runs end-to-end in both execution modes and the
     trained smoke model moves toward goals under replanning;
     aggregate_results.py summarizes the smoke results
  8. the shipped experiment configs and submit files are consistent
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_goal_reaching
import train
import train_multistep
from datasets.structured_dot_world import (
    MotionType,
    StructuredDotWorldDataset,
    StructuredDotWorldSegmentDataset,
    make_combined_config,
    make_coupled_config,
    make_independent_random_config,
    make_independent_static_config,
    step_positions,
)
from datasets.sprite_world import (
    ControlConfig,
    SpriteWorldConfig,
    SpriteWorldDataset,
    SpriteWorldSegmentDataset,
    step_pose,
)
from models import MultistepInverseModel

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "multistep_inverse"

STRUCTURED_CONFIGS = {
    "single_dot": make_independent_static_config(num_independent=1, num_static=0),
    "2_independent": make_independent_static_config(num_independent=2, num_static=0),
    "1_coupled_pair": make_coupled_config(num_pairs=1),
    "1_dot_1_random": make_independent_random_config(num_independent=1, num_random=1),
    "combined": make_combined_config(),
}


# ─────────────────────────────────────────────────────────────────
#  1. k_min=k_max=1 bitwise equivalence with the transition datasets
# ─────────────────────────────────────────────────────────────────

def test_structured_segment_k1_bitwise():
    for name, cfg in STRUCTURED_CONFIGS.items():
        base = StructuredDotWorldDataset(cfg, num_samples=8, seed=7)
        seg = StructuredDotWorldSegmentDataset(cfg, num_samples=8, seed=7, k_min=1, k_max=1)
        for i in range(8):
            obs_t, action, obs_tp1, pos_t = base[i]
            s_obs_t, s_action, s_obs_tp1, s_obs_tpk, s_k, s_pos_t = seg[i]
            assert torch.equal(obs_t, s_obs_t), name
            assert torch.equal(action, s_action), name
            assert torch.equal(obs_tp1, s_obs_tp1), name
            assert torch.equal(pos_t, s_pos_t), name
            assert s_k.item() == 1, name
            assert torch.equal(s_obs_tpk, s_obs_tp1), name       # k=1: o_{t+k} ≡ o_{t+1}


def test_sprite_segment_k1_bitwise():
    for control in (ControlConfig.XY, ControlConfig.XYT):
        cfg = SpriteWorldConfig(control=control)
        base = SpriteWorldDataset(cfg, num_samples=8, seed=3)
        seg = SpriteWorldSegmentDataset(cfg, num_samples=8, seed=3, k_min=1, k_max=1)
        for i in range(8):
            obs_t, action, obs_tp1, state_t = base[i]
            s_obs_t, s_action, s_obs_tp1, s_obs_tpk, s_k, s_state_t = seg[i]
            assert torch.equal(obs_t, s_obs_t), control
            assert torch.equal(action, s_action), control
            assert torch.equal(obs_tp1, s_obs_tp1), control
            assert torch.equal(state_t, s_state_t), control
            assert s_k.item() == 1 and torch.equal(s_obs_tpk, s_obs_tp1), control


# ─────────────────────────────────────────────────────────────────
#  2. random-k segment self-consistency
# ─────────────────────────────────────────────────────────────────

def _structured_action_slices(cfg):
    """(group, dot_range, action_slice) for controllable groups, in order."""
    out, a_idx = [], 0
    for group, (start, end) in zip(cfg.groups, cfg.group_ranges()):
        if group.motion_type is MotionType.INDEPENDENT:
            out.append((group, (start, end), slice(a_idx, a_idx + 2 * group.num_dots)))
            a_idx += 2 * group.num_dots
        elif group.motion_type is MotionType.COUPLED:
            out.append((group, (start, end), slice(a_idx, a_idx + 2 * group.num_pairs)))
            a_idx += 2 * group.num_pairs
    return out


def test_structured_segment_consistency():
    k = 8
    for name, cfg in STRUCTURED_CONFIGS.items():
        seg = StructuredDotWorldSegmentDataset(cfg, num_samples=0, seed=0)
        rng = np.random.default_rng(1)
        for _ in range(10):
            P, A = seg.sample_segment(rng, k)
            assert P.shape == (k + 1, cfg.num_dots, 2) and A.shape == (k, cfg.action_dim), name
            c_lo, c_hi = cfg.dot_radius, cfg.image_size - 1 - cfg.dot_radius
            assert P.min() >= c_lo and P.max() <= c_hi, name
            for t in range(k):
                delta = P[t + 1] - P[t]
                for group, (start, end), a_sl in _structured_action_slices(cfg):
                    if group.motion_type is MotionType.INDEPENDENT:
                        assert np.array_equal(A[t][a_sl], delta[start:end].reshape(-1)), name
                    else:  # COUPLED: both dots of each pair share the recorded (dx, dy)
                        pair = A[t][a_sl].reshape(-1, 2)
                        for p, d in enumerate(pair):
                            assert np.array_equal(delta[start + 2 * p], d), name
                            assert np.array_equal(delta[start + 2 * p + 1], d), name
                for group, (start, end) in zip(cfg.groups, cfg.group_ranges()):
                    if group.motion_type is MotionType.STATIC:
                        assert np.array_equal(delta[start:end], np.zeros_like(delta[start:end])), name

    # __getitem__: k in range, deterministic per index, fields derive from the
    # same per-index segment, first action is the inverse target.
    cfg = STRUCTURED_CONFIGS["combined"]
    seg = StructuredDotWorldSegmentDataset(cfg, num_samples=64, seed=21, k_min=1, k_max=8)
    ks = [seg[i][4].item() for i in range(64)]
    assert set(ks) <= set(range(1, 9)) and len(set(ks)) >= 6, sorted(set(ks))
    assert ks == [seg[i][4].item() for i in range(64)]
    for i in range(6):
        _, action, _, obs_tpk, k_t, pos_t = seg[i]
        rng = np.random.default_rng(21 + i)
        k = int(rng.integers(1, 8, endpoint=True))
        P, A = seg.sample_segment(rng, k)
        assert k_t.item() == k
        assert torch.equal(action, torch.from_numpy(A[0]))
        assert torch.equal(pos_t, torch.from_numpy(P[0].reshape(-1).astype(np.float32)))


def test_sprite_segment_consistency():
    k = 8
    cfg = SpriteWorldConfig(control=ControlConfig.XYT)
    seg = SpriteWorldSegmentDataset(cfg, num_samples=0, seed=0)
    rng = np.random.default_rng(1)
    lo = cfg.bounding_radius + 1
    hi = cfg.image_size - 1 - lo
    for _ in range(10):
        S, A = seg.sample_segment(rng, k)
        assert S.shape == (k + 1, 3) and A.shape == (k, 3)
        assert S[:, :2].min() >= lo and S[:, :2].max() <= hi
        assert (S[:, 2] >= 0).all() and (S[:, 2] < 2 * np.pi).all()
        for t in range(k):
            # actions are stored float32, states float64 → float32 tolerance
            assert np.allclose(S[t + 1, :2] - S[t, :2], A[t, :2], atol=1e-5)
            dtheta = (S[t + 1, 2] - S[t, 2]) - A[t, 2]     # 0 mod 2π up to rounding
            assert abs(np.angle(np.exp(1j * dtheta))) < 1e-5

    # __getitem__ fields derive from the same per-index segment.
    seg = SpriteWorldSegmentDataset(cfg, num_samples=4, seed=21, k_min=1, k_max=8)
    for i in range(4):
        _, action, _, _, k_t, state_t = seg[i]
        rng = np.random.default_rng(21 + i)
        k = int(rng.integers(1, 8, endpoint=True))
        S, A = seg.sample_segment(rng, k)
        assert k_t.item() == k
        assert torch.equal(action, torch.from_numpy(A[0]))
        assert torch.equal(state_t, torch.from_numpy(S[0].astype(np.float32)))


# ─────────────────────────────────────────────────────────────────
#  3. the head is conditioned on k
# ─────────────────────────────────────────────────────────────────

def test_multistep_inverse_model_conditions_on_k():
    torch.manual_seed(0)
    h = MultistepInverseModel(latent_dim=16, action_dim=2, hidden_dim=32, k_max=8)
    z, z_tpk = torch.randn(4, 16), torch.randn(4, 16)
    outs = [h(z, z_tpk, torch.full((4,), k, dtype=torch.int64)) for k in (1, 2, 8)]
    assert all(o.shape == (4, 2) for o in outs)
    assert not torch.allclose(outs[0], outs[1])
    assert not torch.allclose(outs[1], outs[2])
    # same (z, z', k) → same output (no hidden state)
    assert torch.equal(outs[0], h(z, z_tpk, torch.full((4,), 1, dtype=torch.int64)))


# ─────────────────────────────────────────────────────────────────
#  4. loss equivalence at k_max=1
# ─────────────────────────────────────────────────────────────────

def _first_batch(ds, n=16):
    cols = list(zip(*(ds[i] for i in range(n))))
    return tuple(torch.stack(c) for c in cols)


def test_losses_k1_equal():
    device = torch.device("cpu")
    # structured
    cfg = STRUCTURED_CONFIGS["combined"]
    torch.manual_seed(0)
    models = train.build_models({"latent_dim": 16, "hidden_dim": 32}, cfg.image_size,
                                cfg.action_dim, device)
    batch_base = _first_batch(StructuredDotWorldDataset(cfg, num_samples=16, seed=5))
    batch_seg = _first_batch(StructuredDotWorldSegmentDataset(cfg, num_samples=16, seed=5, k_min=1, k_max=1))
    scale = float(cfg.max_displacement)
    l_fwd, l_inv, *_ = train.compute_losses(models, batch_base, scale, True, device)
    m_fwd, m_inv, *_ = train_multistep.compute_losses(models, batch_seg, scale, True, device, k_max=1)
    assert torch.equal(l_fwd, m_fwd) and torch.equal(l_inv, m_inv)

    # sprite (per-DOF action scale)
    scfg = SpriteWorldConfig(control=ControlConfig.XYT)
    torch.manual_seed(0)
    models = train.build_models({"latent_dim": 16, "hidden_dim": 32}, scfg.image_size,
                                scfg.action_dim, device)
    batch_base = _first_batch(SpriteWorldDataset(scfg, num_samples=16, seed=5))
    batch_seg = _first_batch(SpriteWorldSegmentDataset(scfg, num_samples=16, seed=5, k_min=1, k_max=1))
    scale = torch.tensor(scfg.action_scale, dtype=torch.float32)
    l_fwd, l_inv, *_ = train.compute_losses(models, batch_base, scale, True, device)
    m_fwd, m_inv, *_ = train_multistep.compute_losses(models, batch_seg, scale, True, device, k_max=1)
    assert torch.equal(l_fwd, m_fwd) and torch.equal(l_inv, m_inv)


# ─────────────────────────────────────────────────────────────────
#  5. end-to-end: train_multistep.py k_max=1 reproduces train.py bitwise
# ─────────────────────────────────────────────────────────────────

_SMOKE: dict = {}


def _smoke_dir() -> Path:
    if "dir" not in _SMOKE:
        d = Path(tempfile.mkdtemp(prefix="multistep_smoke_"))
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        _SMOKE["dir"] = d
    return _SMOKE["dir"]


def _structured_smoke_cfg(run_name: str, k_max: int | None = None) -> dict:
    cfg = {
        "dataset": {"image_size": 64, "dot_radius": 2, "max_displacement": 16,
                    "allow_overlap": False, "train_samples": 1024, "eval_samples": 128,
                    "groups": [{"motion": "independent", "num_dots": 1}]},
        "model": {"latent_dim": 16, "hidden_dim": 64},
        "training": {"seed": 42, "epochs": 12, "batch_size": 128, "lr": 5.0e-4,
                     "lambda": 10.0, "eval_every": 6},
        "output": {"run_name": run_name},
    }
    if k_max is not None:
        cfg["training"]["horizon_k_min"] = 1
        cfg["training"]["horizon_k_max"] = k_max
    return cfg


def _sprite_smoke_cfg(run_name: str, k_max: int | None = None) -> dict:
    cfg = {
        "dataset": {"world": "sprite", "control": "xyt", "shape": "arrow",
                    "image_size": 64, "sprite_scale": 7.5, "supersample": 4,
                    "max_delta_xy": 8.0, "max_delta_theta": None,
                    "train_samples": 96, "eval_samples": 32},
        "model": {"latent_dim": 16, "hidden_dim": 64},
        "training": {"seed": 42, "epochs": 2, "batch_size": 32, "lr": 5.0e-4,
                     "lambda": 10.0, "eval_every": 2},
        "output": {"run_name": run_name},
    }
    if k_max is not None:
        cfg["training"]["horizon_k_min"] = 1
        cfg["training"]["horizon_k_max"] = k_max
    return cfg


def _run_main(module, config_path: Path) -> None:
    argv = sys.argv
    sys.argv = [module.__name__, "--config", str(config_path)]
    try:
        module.main()
    finally:
        sys.argv = argv


def _train_smoke(module, cfg: dict) -> Path:
    """Train one smoke config under the shared tmp experiment dir; cached."""
    run_name = cfg["output"]["run_name"]
    if run_name in _SMOKE:
        return _SMOKE[run_name]
    exp_dir = _smoke_dir()
    config_path = exp_dir / f"config_{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    _run_main(module, config_path)
    run_dir = exp_dir / "results" / run_name
    _SMOKE[run_name] = run_dir
    return run_dir


def _assert_state_dicts_equal(a_path: Path, b_path: Path) -> None:
    a = torch.load(a_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    assert a.keys() == b.keys()
    for part in a:
        for key in a[part]:
            assert torch.equal(a[part][key], b[part][key]), f"{part}.{key} differs"


def test_train_k1_reproduces_train_py_structured():
    ref = _train_smoke(train, _structured_smoke_cfg("smoke_dot_ref"))
    ms = _train_smoke(train_multistep, _structured_smoke_cfg("smoke_dot_kmax1", k_max=1))
    _assert_state_dicts_equal(ref / "model.pt", ms / "model.pt")
    h_ref = torch.load(ref / "train_history.pt", map_location="cpu", weights_only=False)
    h_ms = torch.load(ms / "train_history.pt", map_location="cpu", weights_only=False)
    assert h_ref["train"] == h_ms["train"]
    for key in ("epoch", "fwd", "inv", "total"):
        assert h_ref["eval"][key] == h_ms["eval"][key]


def test_train_k1_reproduces_train_py_sprite():
    ref = _train_smoke(train, _sprite_smoke_cfg("smoke_sprite_ref"))
    ms = _train_smoke(train_multistep, _sprite_smoke_cfg("smoke_sprite_kmax1", k_max=1))
    _assert_state_dicts_equal(ref / "model.pt", ms / "model.pt")


# ─────────────────────────────────────────────────────────────────
#  6. closed-loop steppers replay recorded actions exactly
# ─────────────────────────────────────────────────────────────────

def test_step_positions_replays_segment():
    for name, cfg in STRUCTURED_CONFIGS.items():
        seg = StructuredDotWorldSegmentDataset(cfg, num_samples=0, seed=0)
        P, A = seg.sample_segment(np.random.default_rng(11), 8)
        ctrl = [d for group, (start, end) in zip(cfg.groups, cfg.group_ranges())
                if group.motion_type in (MotionType.INDEPENDENT, MotionType.COUPLED)
                for d in range(start, end)]
        static = [d for group, (start, end) in zip(cfg.groups, cfg.group_ranges())
                  if group.motion_type is MotionType.STATIC for d in range(start, end)]
        cur = P[0].astype(np.float64)
        for t in range(8):
            cur = step_positions(cfg, cur, A[t].astype(np.float64), np.random.default_rng(99))
        assert np.allclose(cur[ctrl], P[-1][ctrl]), name
        assert np.allclose(cur[static], P[0][static]), name


def test_step_pose_replays_segment():
    cfg = SpriteWorldConfig(control=ControlConfig.XYT)
    seg = SpriteWorldSegmentDataset(cfg, num_samples=0, seed=0)
    S, A = seg.sample_segment(np.random.default_rng(11), 8)
    cur = S[0].copy()
    for t in range(8):
        cur = step_pose(cfg, cur, A[t].astype(np.float64), np.random.default_rng(99))
    # replayed actions are float32 → float32 accumulation tolerance
    assert np.allclose(cur[:2], S[-1][:2], atol=1e-4)
    assert abs(np.angle(np.exp(1j * (cur[2] - S[-1][2])))) < 1e-4


# ─────────────────────────────────────────────────────────────────
#  7. goal-reaching eval end-to-end (both modes) + aggregation
# ─────────────────────────────────────────────────────────────────

def _smoke_eval_cfg() -> dict:
    return {
        "model_runs": ["smoke_dot_kmax1"],
        "eval": {"episodes": 20, "path_steps": 8, "subgoal_spacings": [1, 2],
                 "execution": ["replanning", "open_loop"],
                 "success_threshold": 3.0, "theta_threshold": 0.3,
                 "budget_factor": 3.0, "budget_slack": 2, "seed": 7,
                 "save_traces": True},
    }


def test_eval_goal_reaching_smoke():
    run_dir = _train_smoke(train_multistep, _structured_smoke_cfg("smoke_dot_kmax1", k_max=1))
    exp_dir = _smoke_dir()
    config_path = exp_dir / "config_eval_smoke.yaml"
    config_path.write_text(yaml.safe_dump(_smoke_eval_cfg(), sort_keys=False))
    _run_main(eval_goal_reaching, config_path)

    out = torch.load(run_dir / "goal_reaching.pt", map_location="cpu", weights_only=False)
    assert out["spacings"] == [1, 2] and out["k_max"] == 1
    assert out["modes"] == ["replanning", "open_loop"]
    for mode in out["modes"]:
        for m in (1, 2):
            agg = out["summary"][mode][m]
            assert agg["episodes"] == 20 and 0.0 <= agg["success_rate"] <= 1.0
            assert len(out["episodes"][mode][m]["traces"]) == 20
    # Even the short smoke training must move toward goals under replanning.
    for m in (1, 2):
        agg = out["summary"]["replanning"][m]
        assert agg["pos_err_mean"] < agg["pos_err_init_mean"], (
            f"replanning spacing {m}: final {agg['pos_err_mean']:.2f} px not below "
            f"initial {agg['pos_err_init_mean']:.2f} px")
    # Open-loop executes exactly one action per oracle pair.
    assert out["summary"]["open_loop"][1]["steps_mean"] == 8.0
    assert out["summary"]["open_loop"][2]["steps_mean"] == 4.0


def test_eval_goal_reaching_multistep_head_smoke():
    """A k_max>1 run goes through the k-conditioned decode path end-to-end."""
    cfg = _structured_smoke_cfg("smoke_dot_kmax4", k_max=4)
    cfg["training"]["epochs"] = 4
    run_dir = _train_smoke(train_multistep, cfg)
    hist = torch.load(run_dir / "train_history.pt", map_location="cpu", weights_only=False)
    assert set(hist["eval"]["inv_per_k"][-1]) == {1, 2, 3, 4}       # per-k logging
    exp_dir = _smoke_dir()
    ev = _smoke_eval_cfg()
    ev["model_runs"] = ["smoke_dot_kmax4"]
    ev["eval"]["episodes"] = 5
    config_path = exp_dir / "config_eval_smoke_kmax4.yaml"
    config_path.write_text(yaml.safe_dump(ev, sort_keys=False))
    _run_main(eval_goal_reaching, config_path)
    out = torch.load(run_dir / "goal_reaching.pt", map_location="cpu", weights_only=False)
    assert out["k_max"] == 4 and out["modes"] == ["replanning", "open_loop"]


def test_aggregate_results_smoke():
    _train_smoke(train, _structured_smoke_cfg("smoke_dot_ref"))
    results_dir = _smoke_dir() / "results"
    proc = subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "aggregate_results.py"),
         "--results-dir", str(results_dir)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "smoke_dot_kmax1" in proc.stdout and "smoke_dot_kmax4" in proc.stdout
    assert (results_dir / "summary.csv").exists()


# ─────────────────────────────────────────────────────────────────
#  8. shipped experiment configs and submit files are consistent
# ─────────────────────────────────────────────────────────────────

def test_experiment_configs_parse():
    train_cfgs = sorted(EXPERIMENT_DIR.glob("config_*_kmax*.yaml"))
    eval_cfgs = sorted(EXPERIMENT_DIR.glob("config_eval_*.yaml"))
    assert len(train_cfgs) == 20 and len(eval_cfgs) == 5
    assert not set(train_cfgs) & set(eval_cfgs)          # run.sh globs stay disjoint

    run_names = set()
    for path in train_cfgs:
        cfg = train.load_config(path)
        world_cfg, _, _ = train.build_world(cfg["dataset"])
        assert world_cfg.action_dim > 0, path.name
        k_min = int(cfg["training"]["horizon_k_min"])
        k_max = int(cfg["training"]["horizon_k_max"])
        assert k_min == 1 and k_max in (1, 2, 4, 8), path.name
        run_name = cfg["output"]["run_name"]
        assert path.name == f"config_{run_name}.yaml" and run_name.endswith(f"_kmax{k_max}")
        run_names.add(run_name)

    for path in eval_cfgs:
        cfg = train.load_config(path)
        assert set(cfg["model_runs"]) <= run_names, path.name
        for key in ("episodes", "path_steps", "subgoal_spacings", "execution",
                    "success_threshold", "theta_threshold", "budget_factor",
                    "budget_slack", "seed"):
            assert key in cfg["eval"], f"{path.name}: missing eval.{key}"
        assert set(cfg["eval"]["execution"]) == {"replanning", "open_loop"}, path.name

    # train_ref.sub + train_sweep.sub queue exactly the 20 configs, disjointly.
    def queued(sub_name):
        text = (EXPERIMENT_DIR / sub_name).read_text()
        items = text.split("queue cfg from (", 1)[1].split(")", 1)[0].split()
        for item in items:
            assert (EXPERIMENT_DIR / item).exists(), f"{sub_name}: {item} missing"
        return set(items)

    ref, sweep = queued("train_ref.sub"), queued("train_sweep.sub")
    assert len(ref) == 5 and all("kmax1.yaml" in c for c in ref)
    assert len(sweep) == 15 and not ref & sweep
    assert ref | sweep == {p.name for p in train_cfgs}


# ─────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_structured_segment_k1_bitwise,
    test_sprite_segment_k1_bitwise,
    test_structured_segment_consistency,
    test_sprite_segment_consistency,
    test_multistep_inverse_model_conditions_on_k,
    test_losses_k1_equal,
    test_train_k1_reproduces_train_py_structured,
    test_train_k1_reproduces_train_py_sprite,
    test_step_positions_replays_segment,
    test_step_pose_replays_segment,
    test_eval_goal_reaching_smoke,
    test_eval_goal_reaching_multistep_head_smoke,
    test_aggregate_results_smoke,
    test_experiment_configs_parse,
]


def main() -> None:
    for fn in ALL_TESTS:
        print(f"[ RUN  ] {fn.__name__}")
        fn()
        print(f"[  OK  ] {fn.__name__}")
    print(f"\nAll {len(ALL_TESTS)} checks passed.")


if __name__ == "__main__":
    main()
