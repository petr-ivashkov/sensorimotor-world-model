"""Planning evaluation for latent world model checkpoints.

Loads a model from config.yaml and released weights.pt or a training
checkpoint, and evaluates it with a CEM planner using
stable_worldmodel's World + CEMSolver + WorldModelPolicy.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import hydra
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
    method = train_cfg.get("method", "smwm")
    if method == "dino_wm":
        from baselines.dino_wm.model import build_dino_wm

        model = build_dino_wm(train_cfg, pretrained=False)
    elif method == "pldm":
        from baselines.pldm.planning_model import PlanningPLDM

        model = PlanningPLDM(**{
            name: hydra.utils.instantiate(train_cfg.model[name])
            for name in ("encoder", "predictor", "action_encoder", "projector", "pred_proj")
        })
    else:
        model = build_jepa(train_cfg)
    weights = run_dir / "weights.pt"
    if weights.is_file():
        state = torch.load(weights, map_location="cpu", weights_only=True)
    else:
        ckpt = torch.load(
            run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False
        )
        state = {
            k[len("model.") :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    # strict: a mismatch would leave tensors randomly initialised and still plan.
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, train_cfg


def img_transform(img_size: int, method: str = "smwm"):
    steps = [transforms.ToImage()]
    if method in {"pldm", "dino_wm"}:
        steps.extend([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        ])
    steps.append(transforms.Resize(size=img_size))
    return transforms.Compose(steps)


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


def evaluate_run(run, cfg, out_dir: Path, dataset, episodes, start_steps):
    policy_name = run.get("policy", "cem")

    run_result = {
        "name": run["name"],
        "policy": policy_name,
        "seed": cfg["seed"],
        "task_seed": cfg["eval"].get("task_seed", cfg["seed"]),
        "env": cfg["env"],
        "environment": cfg["env_label"],
        "method": "random",
        "train_seed": None,
        "repeat": cfg["repeat"],
        "inverse_weight": None,
        "sigreg_weight": None,
        "history": None,
        "goal_offset": cfg["eval"]["goal_offset_steps"],
        "eval_budget": cfg["eval"]["eval_budget"],
        "num_eval": len(episodes),
    }
    if policy_name == "random":
        print(f"\n=== Evaluating {run['name']} with random policy ===")
        world = build_world(cfg)
        policy = swm.policy.RandomPolicy(seed=cfg["seed"])
    elif policy_name == "cem":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n=== Loading {run['name']} from {run['run_dir']} ===")
        model, train_cfg = load_jepa_from_run(run["run_dir"], device=device)
        if train_cfg.get("env", cfg["env"]) != cfg["env"]:
            raise ValueError("Evaluation environment differs from the checkpoint's training environment.")
        method = train_cfg.get("method", "smwm")
        run_result.update(
            method=method,
            train_seed=int(train_cfg.seed),
            history=int(train_cfg.wm.get("history_size", 1)),
            inverse_weight=train_cfg.get("loss", {}).get("inverse", {}).get("weight"),
            sigreg_weight=train_cfg.get("loss", {}).get("sigreg", {}).get("weight"),
        )
        world = build_world(cfg, train_cfg=train_cfg)
        process = fit_processors(dataset, cfg["dataset"]["keys_to_cache"])
        transform = {
            "pixels": img_transform(cfg["eval"]["img_size"], method),
            "goal": img_transform(cfg["eval"]["img_size"], method),
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
    run_result["metrics"] = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metrics.items()
    }
    run_result["elapsed_seconds"] = elapsed
    return run_result


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg_node):
    cfg = OmegaConf.to_container(cfg_node, resolve=True)
    if cfg["policy"] == "random":
        name = f"{cfg['env']}_random_repeat{cfg['repeat']}"
        run = {"name": name, "policy": "random"}
        cfg["seed"] += int(cfg["repeat"])
    elif cfg["policy"] == "cem":
        if cfg["run_dir"] is None:
            raise ValueError("Set run_dir to a trained run, or policy=random for the random baseline.")
        run_dir = Path(cfg["run_dir"]).expanduser().resolve()
        if not (run_dir / "config.yaml").is_file():
            raise FileNotFoundError(run_dir / "config.yaml")
        if not any((run_dir / name).is_file() for name in ("weights.pt", "checkpoints/last.ckpt")):
            raise FileNotFoundError(f"Expected weights.pt or checkpoints/last.ckpt in {run_dir}")
        train_cfg = OmegaConf.load(run_dir / "config.yaml")
        cfg["seed"] += int(train_cfg.seed)
        run = {"name": train_cfg.get("subdir") or run_dir.name, "run_dir": str(run_dir)}
    else:
        raise ValueError(f"Unknown policy: {cfg['policy']}")

    goal = int(cfg["eval"]["goal_offset_steps"])
    budget = int(cfg["eval"]["eval_budget"])
    if cfg["plan_config"]["horizon"] * cfg["plan_config"]["action_block"] > budget:
        raise ValueError("Planning horizon in primitive steps exceeds the evaluation budget.")
    run_out = Path(cfg["output_dir"]) / run["name"] / f"goal{goal}_budget{budget}"
    if run_out.exists() and any(run_out.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {run_out}. Choose another output_dir.")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    dataset = swm.data.HDF5Dataset(
        cfg["eval"]["dataset_name"],
        keys_to_cache=cfg["dataset"]["keys_to_cache"],
        cache_dir=cfg["data_dir"],
    )
    task_seed = int(cfg["eval"].get("task_seed", cfg["seed"]))
    print(f"Sampling evaluation tasks with seed={task_seed}.")
    episodes, start_steps = sample_start_indices(
        dataset,
        configured_num_envs(cfg),
        cfg["eval"]["goal_offset_steps"],
        task_seed,
    )

    run_out.mkdir(parents=True, exist_ok=True)
    result = evaluate_run(run, cfg, run_out, dataset, episodes, start_steps)
    result["evaluation_config"] = cfg
    path = run_out / "metrics.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
