"""Planning evaluation for latent world model checkpoints.

Loads one or more trained JEPA runs (rebuilt from their saved config.yaml
and Lightning last.ckpt) and evaluates each with a CEM planner using
stable_worldmodel's World + CEMSolver + WorldModelPolicy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm

REPO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from jepa import JEPA
from module import ARPredictor, Embedder, InverseModel, MLP
from utils import load_composed_config


def build_jepa(cfg):
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=int(cfg.wm.get("history_size", 1)),
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        ),
        action_encoder=Embedder(input_dim=effective_act_dim, emb_dim=embed_dim),
        projector=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        ),
        inverse_model=InverseModel(
            embed_dim=embed_dim,
            action_dim=effective_act_dim,
            hidden_dim=cfg.inverse.get("hidden_dim", 256),
        ),
    )


def load_jepa_from_run(run_dir: Path, device: str = "cuda"):
    run_dir = Path(run_dir)
    train_cfg = OmegaConf.load(run_dir / "config.yaml")
    model = build_jepa(train_cfg)
    ckpt = torch.load(
        run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False
    )
    state = {
        k[len("model.") :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[{run_dir.name}] load_state_dict: missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, train_cfg


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(size=img_size),
        ]
    )


def fit_processors(dataset, keys_to_cache):
    process = {}
    for col in keys_to_cache:
        if col == "pixels":
            continue
        proc = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        proc.fit(data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc
    return process


def configured_num_envs(cfg):
    return int(cfg.get("world", {}).get("num_envs", cfg["eval"]["num_eval"]))


def build_world(cfg, train_cfg=None):
    world_cfg = dict(cfg["world"])
    env_name = world_cfg.pop("env_name")
    num_envs = int(world_cfg.pop("num_envs", cfg["eval"]["num_eval"]))
    frame_skip = int(world_cfg.pop("frame_skip", 1))
    train_history_size = 1
    if train_cfg is not None and "wm" in train_cfg:
        train_history_size = int(train_cfg.wm.get("history_size", 1) or 1)
    model_history_size = int(world_cfg.pop("history_size", train_history_size))
    action_block = int(cfg.get("plan_config", {}).get("action_block", 1))
    default_wrapper_history = (
        (model_history_size - 1) * action_block + 1
        if model_history_size > 1
        else 1
    )
    wrapper_history_size = int(
        world_cfg.pop("wrapper_history_size", default_wrapper_history)
    )
    eval_budget = int(cfg["eval"]["eval_budget"])
    max_episode_steps = int(world_cfg.pop("max_episode_steps", 2 * eval_budget))
    if max_episode_steps < eval_budget:
        raise ValueError(
            "world.max_episode_steps must be >= eval.eval_budget "
            f"(got {max_episode_steps} < {eval_budget})"
        )
    seed = int(world_cfg.pop("seed", cfg.get("seed", 2349867)))

    img_size = int(cfg["eval"].get("img_size", 224))
    image_height = int(world_cfg.get("height", img_size))
    image_width = int(world_cfg.get("width", img_size))

    return swm.World(
        env_name=env_name,
        num_envs=num_envs,
        image_shape=(image_height, image_width),
        seed=seed,
        history_size=wrapper_history_size,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        **world_cfg,
    )


def sample_start_indices(dataset, num_eval, goal_offset_steps, seed):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_ids_all = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    ep_unique = np.unique(ep_ids_all)
    ep_len = {int(e): int(step_idx[ep_ids_all == e].max()) + 1 for e in ep_unique}
    max_start_per_row = np.array(
        [ep_len[int(e)] - goal_offset_steps - 1 for e in ep_ids_all]
    )
    valid = np.nonzero(step_idx <= max_start_per_row)[0]
    print(f"{len(valid)} valid starting points found for evaluation.")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(valid) - 1, size=num_eval, replace=False)
    picks = np.sort(valid[picks])
    rows = dataset.get_row_data(picks)
    return rows[col_name].tolist(), rows["step_idx"].tolist()


def collect_results(out_root: Path, preferred_order=None):
    results_by_name = {}
    for metrics_path in sorted(out_root.glob("*/metrics.json")):
        try:
            result = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid metrics file {metrics_path}: {exc}")
            continue
        name = result.get("name", metrics_path.parent.name)
        results_by_name[name] = result

    preferred_order = preferred_order or []
    ordered_names = [name for name in preferred_order if name in results_by_name]
    ordered_names.extend(
        sorted(name for name in results_by_name if name not in set(ordered_names))
    )
    return [results_by_name[name] for name in ordered_names]


def evaluate_run(run, cfg, out_dir: Path, dataset, episodes, start_steps):
    policy_name = run.get("policy", "cem")

    run_result = {
        "name": run["name"],
        "policy": policy_name,
        "seed": cfg["seed"],
        "task_seed": cfg["eval"].get("task_seed", cfg["seed"]),
    }
    if policy_name == "random":
        print(f"\n=== Evaluating {run['name']} with random policy ===")
        world = build_world(cfg)
        policy = swm.policy.RandomPolicy(seed=cfg["seed"])
    elif policy_name == "cem":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== Loading {run['name']} from {run['run_dir']} ===")
        model, train_cfg = load_jepa_from_run(run["run_dir"], device=device)
        world = build_world(cfg, train_cfg=train_cfg)
        process = fit_processors(dataset, cfg["dataset"]["keys_to_cache"])
        transform = {
            "pixels": img_transform(cfg["eval"]["img_size"]),
            "goal": img_transform(cfg["eval"]["img_size"]),
        }
        solver = swm.solver.CEMSolver(model=model, seed=cfg["seed"], **cfg["solver"])
        plan_cfg = swm.PlanConfig(**cfg["plan_config"])
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=plan_cfg, process=process, transform=transform
        )
        run_result["run_dir"] = str(run["run_dir"])
    else:
        raise ValueError(
            f"Unsupported run policy '{policy_name}' for run '{run['name']}'"
        )

    world.set_policy(policy)

    t0 = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=start_steps,
        goal_offset_steps=cfg["eval"]["goal_offset_steps"],
        eval_budget=cfg["eval"]["eval_budget"],
        episodes_idx=episodes,
        callables=cfg["eval"]["callables"],
        save_video=cfg["eval"].get("save_video", True),
        video_path=str(out_dir),
    )
    elapsed = time.time() - t0
    print(f"[{run['name']}] metrics: {metrics}  ({elapsed:.1f}s)")
    run_result["metrics"] = metrics
    run_result["elapsed_seconds"] = elapsed
    return run_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to experiment config.yaml")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="optional OmegaConf dotlist overrides, e.g. world.history_size=3",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg_node = load_composed_config(config_path)
    overrides = [override for override in args.overrides if override != "--"]
    if overrides:
        OmegaConf.set_struct(cfg_node, False)
        cfg_node = OmegaConf.merge(cfg_node, OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.to_container(cfg_node, resolve=True)

    default_runs_root = config_path.parent / "results"
    out_root = Path(os.environ.get("RUNS_ROOT", str(default_runs_root)))
    out_root.mkdir(parents=True, exist_ok=True)

    missing = []
    for run in cfg["runs"]:
        policy_name = run.get("policy", "cem")
        if policy_name == "random":
            continue
        if policy_name != "cem":
            raise ValueError(
                f"Unsupported run policy '{policy_name}' for run '{run['name']}'"
            )
        rd = Path(run["run_dir"])
        if not (rd / "config.yaml").is_file():
            missing.append(f"  - {run['name']}: missing {rd / 'config.yaml'}")
        if not (rd / "checkpoints" / "last.ckpt").is_file():
            missing.append(f"  - {run['name']}: missing {rd / 'checkpoints' / 'last.ckpt'}")
    if missing:
        raise FileNotFoundError(
            "Preflight failed — the following checkpoints/configs are not present "
            "on this filesystem:\n" + "\n".join(missing)
        )
    print(f"Preflight OK — {len(cfg['runs'])} runs resolved.")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    dataset = swm.data.HDF5Dataset(
        cfg["eval"]["dataset_name"],
        keys_to_cache=cfg["dataset"]["keys_to_cache"],
    )
    task_seed = int(cfg["eval"].get("task_seed", cfg["seed"]))
    print(f"Sampling evaluation tasks with seed={task_seed}.")
    episodes, start_steps = sample_start_indices(
        dataset,
        configured_num_envs(cfg),
        cfg["eval"]["goal_offset_steps"],
        task_seed,
    )

    for run in cfg["runs"]:
        run_out = out_root / run["name"]
        run_out.mkdir(parents=True, exist_ok=True)
        result = evaluate_run(run, cfg, run_out, dataset, episodes, start_steps)
        (run_out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))

    summary_path = out_root / "summary.json"
    all_results = collect_results(
        out_root, preferred_order=[run["name"] for run in cfg["runs"]]
    )
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nWrote summary to {summary_path}")

    print("\n=== Planning eval summary ===")
    for r in all_results:
        print(f"  {r['name']}: {r['metrics']}")


if __name__ == "__main__":
    main()
