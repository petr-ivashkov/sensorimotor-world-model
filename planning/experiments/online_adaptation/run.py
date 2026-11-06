"""Run one resumable collect -> adapt -> diagnose -> plan experiment on one GPU.

All writes stay beneath a new output directory. Existing source models and data
are read in place. --preflight checks prerequisites without creating outputs or
stepping a simulator.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for directory in (ROOT, ROOT / "planning"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("REPO_ROOT", str(ROOT / "planning"))

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from planning.experiments.online_adaptation.adaptation import OnlineLearner
from planning.experiments.online_adaptation.data import (
    ExperienceDataset, action_statistics, episode_metadata, file_digest,
    fixed_tasks, split_episodes,
)
from planning.experiments.online_adaptation.diagnostics import evaluate_diagnostics
from planning.experiments.online_adaptation.environments import (
    FixedActionProcessor, calibrate_shift, collect_round, evaluate_planning,
)


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_finite_json(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _file_identity(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def load_config(path, _parents=()):
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("Configuration base cycle")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    base = value.pop("base", None)
    if base:
        inherited = load_config(path.parent / base, _parents + (path,))
        value = OmegaConf.to_container(OmegaConf.merge(inherited, value), resolve=True)
    return value


def implementation_identity():
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "planning" / name for name in ("jepa.py", "module.py", "eval.py", "utils.py")]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


@contextmanager
def isolated_rng():
    """Keep acquisition/evaluation from changing subsequent dropout/update RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def validate_config(cfg):
    if cfg["initialization"] not in {"full", "encoder_only", "scratch"}:
        raise ValueError("initialization must be full, encoder_only, or scratch")
    if not isinstance(cfg["rounds"], int) or cfg["rounds"] < 0:
        raise ValueError("rounds must be nonnegative; zero evaluates the frozen initialization")
    for key in ("episodes_per_round", "updates_per_round"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["training"]["batch_size"] < 2 or cfg["training"]["num_workers"] < 0:
        raise ValueError("Training needs batch_size >= 2 and nonnegative num_workers")
    fraction = cfg["replay"]["source_fraction"]
    recent = cfg["replay"]["recent_rounds"]
    if not 0 <= fraction < 1 or (recent is not None and (not isinstance(recent, int) or recent < 1)):
        raise ValueError("Invalid replay source_fraction or recent_rounds")
    diagnostic = cfg["diagnostics"]
    if (diagnostic["holdout_episodes"] < 4 or diagnostic["max_clips"] < 2
            or diagnostic["batch_size"] < 1 or not diagnostic["horizons"]
            or min(diagnostic["horizons"]) < 1):
        raise ValueError("Invalid diagnostic episode/sample/batch/horizon settings")
    if cfg["planning"]["enabled"] and cfg["planning"]["num_tasks"] < 1:
        raise ValueError("Planning needs at least one held-out task")
    if int(cfg["planning"]["plan"]["action_block"]) != 5:
        raise ValueError("Planner action_block must match source frameskip=5")
    if int(cfg["planning"]["plan"].get("history_len", 1)) != 1:
        raise ValueError("Initial pilot requires planner history_len=1")
    if cfg["collection"]["policy"] not in {"weak", "random", "cem"}:
        raise ValueError("collection.policy must be weak, random, or cem")
    if cfg.get("source_evaluation", "expert_dataset") not in {"expert_dataset", "matched_collection"}:
        raise ValueError("source_evaluation must be expert_dataset or matched_collection")


def preflight(cfg, source_run, *, source_data=None, source_eval=None, device="cuda"):
    """Read-only package/source-asset inventory. No simulator or model loading."""
    validate_config(cfg)
    source_run = Path(source_run).expanduser().resolve()
    config_path, checkpoint = source_run / "config.yaml", source_run / "checkpoints/last.ckpt"
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Required pretrained artifact is missing: {path}")
    source_cfg = OmegaConf.load(config_path)
    dataset_cfg = source_cfg.data.dataset
    if int(source_cfg.wm.get("history_size", 1)) != 1 or int(dataset_cfg.frameskip) != 5:
        raise ValueError("Initial pilot requires a source checkpoint with H=1, frameskip=5")
    if int(source_cfg.wm.action_dim) != 2:
        raise ValueError("Initial pilot requires Push-T's two-dimensional raw actions")
    if int(source_cfg.img_size) != int(cfg["source_env"].get("img_size", 224)):
        raise ValueError("Environment image size must match the source checkpoint")
    configured = Path(str(dataset_cfg.get("cache_dir", ROOT / "planning/data/external")))
    source_data = Path(source_data) if source_data else configured / f"{dataset_cfg.name}.h5"
    if not source_eval:
        if not source_data.stem.endswith("_train"):
            raise ValueError("Pass --source-eval when source dataset does not end in _train")
        source_eval = source_data.with_name(source_data.stem[:-6] + "_eval.h5")
    source_data, source_eval = Path(source_data).expanduser().resolve(), Path(source_eval).expanduser().resolve()
    if source_data == source_eval:
        raise ValueError("Source training and evaluation datasets must differ")
    inventory = {}
    for label, path in (("source_train", source_data), ("source_eval", source_eval)):
        lengths, _ = episode_metadata(path)
        inventory[label] = {**_file_identity(path), "episodes": len(lengths), "rows": int(lengths.sum())}
    versions = {}
    for name in ("stable-worldmodel", "stable-pretraining", "torch", "transformers", "pymunk", "h5py"):
        versions[name] = importlib.metadata.version(name)
    problems = []
    if versions["stable-worldmodel"] != "0.0.6":
        problems.append("stable-worldmodel must be 0.0.6 for the verified collection API")
    if int(versions["pymunk"].split(".")[0]) < 7:
        problems.append("Push-T needs pymunk >= 7 (Space.on_collision)")
    if versions["transformers"] != "5.5.0":
        problems.append("transformers must match checkpoint training version 5.5.0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        problems.append("CUDA is unavailable")
    report = {"python": platform.python_version(), "versions": versions,
              "source_run": str(source_run), "source_config_sha256": file_digest(config_path),
              "checkpoint": _file_identity(checkpoint), **inventory,
              "device": device, "problems": problems}
    return report, source_cfg


def _load_model(source_run, source_cfg, initialization, device):
    # Existing constructor/strict source weight loader; no source optimizer state.
    from eval import build_jepa, load_jepa_from_run
    if initialization == "full":
        return load_jepa_from_run(source_run, device=device)[0]
    model = build_jepa(source_cfg)
    if initialization == "encoder_only":
        pretrained, _ = load_jepa_from_run(source_run, device="cpu")
        model.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=True)
        model.projector.load_state_dict(pretrained.projector.state_dict(), strict=True)
        del pretrained
    return model.to(device)


def _loader(dataset, *, batch_size, workers=0, seed=0, train=False, source_fraction=0.0, target_size=None):
    generator = torch.Generator().manual_seed(seed)
    kwargs = {"batch_size": batch_size, "num_workers": workers,
              "drop_last": train, "pin_memory": torch.cuda.is_available(),
              "generator": generator}
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    if source_fraction:
        source_size = len(dataset) - target_size
        if source_size < 1 or target_size < 1:
            raise ValueError("Source replay requires nonempty target and source clips")
        weights = torch.cat((torch.full((target_size,), (1-source_fraction)/target_size),
                             torch.full((source_size,), source_fraction/source_size)))
        kwargs["sampler"] = WeightedRandomSampler(weights, max(batch_size, len(dataset)),
                                                  replacement=True, generator=generator)
    else:
        kwargs["shuffle"] = train
    if not len(dataset) or (train and len(dataset) < batch_size and not source_fraction):
        raise ValueError(f"Need at least {batch_size if train else 1} valid clips; found {len(dataset)}")
    return DataLoader(dataset, **kwargs)


def _collect_shard(directory, env, cfg, *, seed, episodes, model=None, stats=None):
    """Commit HDF5 plus provenance together by atomically renaming a directory."""
    directory = Path(directory)
    specification = {"environment": env, "policy": cfg["collection"], "seed": seed, "episodes": episodes}
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["specification"] != specification or file_digest(directory / "experience.h5") != manifest["sha256"]:
            raise ValueError(f"Recorded shard/provenance mismatch: {directory}")
        return directory / "experience.h5", manifest["collection"]
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".collect-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        solver_cfg = {**cfg["planning"]["solver"], "device": str(next(model.parameters()).device) if model is not None else "cpu"}
        with isolated_rng():
            result = collect_round(
                staging / "experience.h5", env, seed=seed, episodes=episodes,
                policy_config=cfg["collection"], model=model,
                process={"action": FixedActionProcessor(stats)} if stats else None,
                plan_config=cfg["planning"]["plan"], solver_config=solver_cfg,
            )
        # Strip temporary paths; only committed paths belong in provenance.
        result["path"] = str(directory / "experience.h5")
        write_json(staging / "manifest.json", {"specification": specification,
                   "collection": result, "sha256": file_digest(staging / "experience.h5")})
        os.replace(staging, directory)
    return directory / "experience.h5", result


def _diagnostic_loaders(path, stats, cfg, split, *, sample_seed):
    settings = cfg["diagnostics"]
    datasets = [ExperienceDataset(path, stats, frameskip=5, num_steps=max(settings["horizons"])+1,
                                 episodes=split[name], max_clips=settings["max_clips"],
                                 sample_seed=sample_seed + offset)
                for offset, name in enumerate(("fit", "test"))]
    loaders = [_loader(dataset, batch_size=settings["batch_size"]) for dataset in datasets]
    return loaders


def _evaluate(learner, loaders, reference, *, path, env, tasks, cfg, stats, device):
    with isolated_rng():
        try:
            diagnostics, reference = evaluate_diagnostics(
                learner.model, *loaders, reference=reference, device=device,
                horizons=cfg["diagnostics"]["horizons"], ridge_alpha=cfg["diagnostics"]["ridge_alpha"],
            )
        finally:
            # SWM's planner uses swmr=True. HDF5 forbids opening the same file
            # with that flag while our ordinary diagnostic readers remain open.
            # These zero-worker loaders reopen lazily at the next snapshot.
            for loader in loaders:
                loader.dataset.close()
        planning = {}
        if cfg["planning"]["enabled"]:
            settings = cfg["planning"]
            planning = evaluate_planning(
                learner.model, path, env, stats, tasks, seed=cfg["seed"] + 500,
                plan_config=settings["plan"], solver_config=settings["solver"],
                eval_budget=settings["eval_budget"], device=device,
            )
    return {"diagnostics": diagnostics, "planning": planning}, reference


def run_experiment(cfg, source_run, output_root, *, source_data=None, source_eval=None,
                   device="cuda", resume=False):
    report, source_cfg = preflight(cfg, source_run, source_data=source_data,
                                   source_eval=source_eval, device=device)
    if report["problems"]:
        raise RuntimeError("Preflight failed: " + "; ".join(report["problems"]))
    source_run, output = Path(source_run).resolve(), Path(output_root).resolve()
    if output == source_run or source_run.is_relative_to(output) or output.is_relative_to(source_run):
        raise ValueError("Online outputs must be separate from the pretrained run")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Nonempty output directory; use --resume for the same run: {output}")
    if resume and not (output / "manifest.json").is_file():
        raise FileNotFoundError("--resume requires the original manifest.json")
    source_path = Path(report["source_train"]["path"])
    source_eval_path = Path(report["source_eval"]["path"])
    manifest = {"config": cfg, "preflight": report, "implementation": implementation_identity(),
                "source_checkpoint_sha256": file_digest(source_run / "checkpoints/last.ckpt")}
    if resume:
        previous = json.loads((output / "manifest.json").read_text())
        if previous != manifest:
            raise ValueError("Source assets, package versions, or run configuration changed; use a new output directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", manifest)
        OmegaConf.save(source_cfg, output / "source_config.yaml", resolve=True)
    print(f"Online adaptation outputs: {output}", flush=True)
    stats_path = output / "action_statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    else:
        stats = action_statistics(source_path)
        write_json(stats_path, stats)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    learner = OnlineLearner(_load_model(source_run, source_cfg, cfg["initialization"], device),
                            source_cfg, device=device, seed=seed, **{
                                key: value for key, value in cfg["training"].items()
                                if key not in {"batch_size", "num_workers"}})
    checkpoint_path = output / "online_state.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.exists() else None
    if checkpoint is not None:
        for name, digest in checkpoint.get("artifacts", {}).items():
            if not (output / name).is_file() or file_digest(output / name) != digest:
                raise ValueError(f"Resume artifact missing or changed: {name}")
        if not (output / "anchor_references.pt").is_file():
            raise ValueError("Resume requires the original anchor_references.pt baseline")
        learner.load_state_dict(checkpoint["learner"])
    source_env = copy.deepcopy(cfg["source_env"])
    target_env = {**copy.deepcopy(source_env), **copy.deepcopy(cfg["target_change"])}
    calibration_path = output / "calibration.json"
    if not calibration_path.exists():
        with isolated_rng():
            calibration = calibrate_shift(source_env, target_env, seed=seed,
                                          steps=cfg["calibration"]["steps"],
                                          policy_config={"policy": "weak", "dist_constraint": cfg["collection"].get("dist_constraint", 100)})
        write_json(calibration_path, calibration)
    calibration = json.loads(calibration_path.read_text())
    if (not calibration["has_contacts"] or
            calibration["mean_block_position_divergence"] < cfg["calibration"]["min_mean_position_divergence"]):
        raise RuntimeError("Selected dynamics change produced too little divergence; inspect calibration.json and choose a new config/output")
    # Evaluation data uses a fixed collector, independent of adaptation/arm.
    holdout_cfg = {**cfg, "collection": {"policy": "weak", "dist_constraint": cfg["collection"].get("dist_constraint", 100)}}
    target_path, _ = _collect_shard(output / "data/target_holdout", target_env, holdout_cfg,
                                   seed=seed+10000, episodes=cfg["diagnostics"]["holdout_episodes"])
    matched_source = cfg.get("source_evaluation", "expert_dataset") == "matched_collection"
    if matched_source:
        source_eval_path, _ = _collect_shard(output / "data/source_holdout", source_env, holdout_cfg,
                                            seed=seed+10000, episodes=cfg["diagnostics"]["holdout_episodes"])
    task_path = output / "evaluation_tasks.json"
    if task_path.exists():
        task_manifest = json.loads(task_path.read_text())
    else:
        target_split = split_episodes(target_path, seed=seed+101)
        source_split = split_episodes(source_eval_path, seed=seed+101 if matched_source else seed+102)
        task_manifest = {"target_split": target_split, "source_split": source_split,
                         "datasets": {"target": str(target_path), "source": str(source_eval_path)}}
        for name, path, split in (("target", target_path, target_split), ("source", source_eval_path, source_split)):
            task_manifest[name] = fixed_tasks(path, episodes=split["test"], count=cfg["planning"]["num_tasks"],
                goal_offset_steps=cfg["planning"]["goal_offset_steps"], seed=seed+103,
                min_block_displacement=cfg["planning"].get("min_block_displacement", 0.0)) if cfg["planning"]["enabled"] else {}
        write_json(task_path, task_manifest)
    loaders = {name: _diagnostic_loaders(path, stats, cfg, task_manifest[f"{name}_split"], sample_seed=seed+200)
               for name, path in (("target", target_path), ("source", source_eval_path))}
    references_path = output / "anchor_references.pt"
    references = torch.load(references_path, weights_only=False, map_location="cpu") if references_path.exists() else {"target": None, "source": None}
    completed = checkpoint["round"] if checkpoint else -1
    experience = checkpoint["experience_transitions"] if checkpoint else 0
    shard_paths = []
    for prior in range(1, completed+1):
        directory = output / f"data/round_{prior:03d}"
        if not directory.is_dir():
            raise ValueError(f"Completed round's experience is missing: {directory}")
        path, _ = _collect_shard(directory, target_env, cfg, seed=seed+prior*1000,
                                 episodes=cfg["episodes_per_round"], stats=stats)
        shard_paths.append(path)
    started = time.monotonic()
    for index in range(completed+1, cfg["rounds"]+1):
        round_start = time.monotonic()
        training, collection, timing = {}, {}, {}
        if index:
            collect_start = time.monotonic()
            path, collection = _collect_shard(output / f"data/round_{index:03d}", target_env, cfg,
                seed=seed+index*1000, episodes=cfg["episodes_per_round"], model=learner.model, stats=stats)
            timing["collection_seconds"] = time.monotonic() - collect_start
            experience += int(collection["raw_transitions"])
            shard_paths.append(path)
            recent = cfg["replay"]["recent_rounds"]
            paths = shard_paths[-recent:] if recent else shard_paths
            datasets = [ExperienceDataset(path, stats, frameskip=5, num_steps=2) for path in paths]
            target_size = sum(map(len, datasets))
            if cfg["replay"]["source_fraction"]:
                datasets.append(ExperienceDataset(source_path, stats, frameskip=5, num_steps=2))
            replay = ConcatDataset(datasets)
            loader = _loader(replay, batch_size=cfg["training"]["batch_size"], workers=cfg["training"]["num_workers"],
                seed=seed+index, train=True, source_fraction=cfg["replay"]["source_fraction"], target_size=target_size)
            update_start = time.monotonic()
            training = learner.update(loader, cfg["updates_per_round"])
            timing["update_seconds"] = time.monotonic() - update_start
            timing["updates_per_second"] = cfg["updates_per_round"] / max(timing["update_seconds"], 1e-9)
            del loader
            for dataset in datasets:
                dataset.close()
        results = {}
        eval_start = time.monotonic()
        for name, path, env in (("target", target_path, target_env), ("source", source_eval_path, source_env)):
            results[name], references[name] = _evaluate(learner, loaders[name], references[name], path=path,
                env=env, tasks=task_manifest[name], cfg=cfg, stats=stats, device=device)
        timing["evaluation_seconds"] = time.monotonic() - eval_start
        timing["round_seconds"] = time.monotonic() - round_start
        timing["session_seconds"] = time.monotonic() - started
        record = {"round": index, "experience_transitions": experience, "updates": learner.update_steps,
                  **results, "training": training, "collection": collection, "timing": timing}
        write_json(output / f"metrics/round_{index:03d}.json", record)
        if not references_path.exists():
            save_torch(references_path, references)
        snapshot_dir = output / f"snapshots/round_{index:03d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(source_cfg, snapshot_dir / "config.yaml", resolve=True)
        state = learner.state_dict()
        save_torch(snapshot_dir / "checkpoints/last.ckpt", {"state_dict": state["state_dict"]})
        artifact_names = ("action_statistics.json", "evaluation_tasks.json", "anchor_references.pt", "calibration.json")
        save_torch(checkpoint_path, {"round": index, "experience_transitions": experience, "learner": state,
                                   "artifacts": {name: file_digest(output / name) for name in artifact_names}})
        print(f"round={index} experience={experience} updates={learner.update_steps} "
              f"target_CKA={results['target']['diagnostics'].get('representation.linear_cka')}", flush=True)
    for pair in loaders.values():
        for loader in pair:
            loader.dataset.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.yaml"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-data", type=Path)
    parser.add_argument("--source-eval", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.preflight:
        report, _ = preflight(cfg, args.source_run, source_data=args.source_data,
                              source_eval=args.source_eval, device=args.device)
        print(json.dumps(_finite_json(report), indent=2, allow_nan=False))
        return 1 if report["problems"] else 0
    if args.output_root is None:
        parser.error("--output-root is required for an experiment")
    run_experiment(cfg, args.source_run, args.output_root, source_data=args.source_data,
                   source_eval=args.source_eval, device=args.device, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
