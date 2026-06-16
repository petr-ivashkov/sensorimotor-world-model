#!/usr/bin/env python3
"""Generate extended-goal-offset planning-eval configs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


EXPERIMENT_NAME = "horizon_sweep"
TRAINING_EXPERIMENT = "train"
SEED = 0
GOAL_BUDGET_PAIRS = (
    (25, 50),
    (40, 80),
    (55, 110),
    (70, 140),
    (85, 170),
    (100, 200),
)
NUM_EVAL = 100
BASE_TASK_SEED = 62025
BASE_POLICY_SEED = 72025


@dataclass(frozen=True)
class Environment:
    label: str
    slug: str
    eval_config: str
    dataset_file: str


ENVIRONMENTS = (
    Environment("TwoRoom", "tworoom", "tworoom", "tworoom_eval.h5"),
    Environment("Reacher", "reacher", "reacher", "reacher_eval.h5"),
    Environment("Push-T", "pusht", "pusht", "pusht_expert_eval.h5"),
    Environment(
        "OGBench-Cube",
        "ogbcube",
        "ogbcube",
        "cube_single_expert_eval.h5",
    ),
)

LEARNED_METHODS = ("forward_only", "inverse", "sigreg")
METHODS = (*LEARNED_METHODS, "random")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_training_manifest(path: Path) -> dict[tuple[str, str, int], str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing final-training manifest: {path}. "
            "Run experiments/train/generate_configs.py first."
        )

    runs: dict[tuple[str, str, int], str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["environment"], row["method"], int(row["seed"]))
            if key in runs:
                raise ValueError(f"Duplicate final-training manifest key: {key}")
            runs[key] = row["run_name"]
    return runs


def offset_label(goal_offset: int) -> str:
    return f"offset_{goal_offset}"


def task_seed(env_idx: int, offset_idx: int) -> int:
    # One fixed task set per environment/offset for all methods.
    return BASE_TASK_SEED + 1000 * env_idx + offset_idx


def policy_seed(env_idx: int, offset_idx: int) -> int:
    return BASE_POLICY_SEED + 1000 * env_idx + offset_idx


def config_for_job(
    env: Environment,
    env_idx: int,
    method: str,
    goal_offset: int,
    eval_budget: int,
    offset_idx: int,
    training_runs: dict[tuple[str, str, int], str],
) -> tuple[str, dict]:
    label = offset_label(goal_offset)
    job_name = f"{env.slug}_{method}_{label}"
    is_random = method == "random"

    run: dict[str, object] = {"name": label}
    final_training_run = None
    if is_random:
        run["policy"] = "random"
    else:
        final_training_run = training_runs[(env.label, method, SEED)]
        run["run_dir"] = (
            "${oc.env:REPO_ROOT}/experiments/"
            f"{TRAINING_EXPERIMENT}/results/{final_training_run}"
        )

    cfg = {
        "defaults": [
            "/eval/base",
            f"/eval/env/{env.eval_config}",
            {"override hydra/job_logging": "disabled"},
            {"override hydra/hydra_logging": "disabled"},
            "_self_",
        ],
        "hydra": {
            "searchpath": ["file://${oc.env:REPO_ROOT}/config"],
        },
        "seed": policy_seed(env_idx, offset_idx),
        "runs": [run],
        "eval": {
            "num_eval": NUM_EVAL,
            "goal_offset_steps": goal_offset,
            "eval_budget": eval_budget,
            "task_seed": task_seed(env_idx, offset_idx),
            "save_video": False,
        },
        "horizon_sweep": {
            "environment": env.label,
            "env_slug": env.slug,
            "method": method,
            "seed": SEED if not is_random else None,
            "random_repeat": 0 if is_random else None,
            "goal_offset": goal_offset,
            "eval_budget": eval_budget,
            "num_eval": NUM_EVAL,
            "task_seed": task_seed(env_idx, offset_idx),
            "policy_seed": policy_seed(env_idx, offset_idx),
            "run_label": label,
            "result_dir": f"results/{env.slug}/{method}/{label}",
            "final_training_run": final_training_run,
        },
    }
    return job_name, cfg


def generate(output_dir: Path, selected_job: str | None = None) -> list[dict[str, str]]:
    root = repo_root()
    training_manifest = (
        root / "experiments" / TRAINING_EXPERIMENT / "generated_configs" / "manifest.tsv"
    )
    training_runs = load_training_manifest(training_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for env_idx, env in enumerate(ENVIRONMENTS):
        for offset_idx, (goal_offset, eval_budget) in enumerate(GOAL_BUDGET_PAIRS):
            for method in METHODS:
                job_name, cfg = config_for_job(
                    env,
                    env_idx,
                    method,
                    goal_offset,
                    eval_budget,
                    offset_idx,
                    training_runs,
                )
                if selected_job and job_name != selected_job:
                    continue

                config_path = output_dir / f"{job_name}.yaml"
                OmegaConf.save(config=OmegaConf.create(cfg), f=config_path)

                metadata = cfg["horizon_sweep"]
                rows.append(
                    {
                        "job_name": job_name,
                        "env": env.slug,
                        "env_label": env.label,
                        "method": method,
                        "seed": str(SEED if method != "random" else ""),
                        "random_repeat": str(0 if method == "random" else ""),
                        "goal_offset": str(goal_offset),
                        "eval_budget": str(eval_budget),
                        "num_tasks": str(NUM_EVAL),
                        "eval_task_seed": str(cfg["eval"]["task_seed"]),
                        "policy_seed": str(cfg["seed"]),
                        "run_label": str(metadata["run_label"]),
                        "result_dir": str(metadata["result_dir"]),
                        "config_path": str(config_path),
                        "dataset_file": env.dataset_file,
                        "final_training_run": str(metadata["final_training_run"] or ""),
                    }
                )

    if not rows:
        raise SystemExit(f"No extended-offset planning jobs matched {selected_job!r}")

    return rows


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = [
        "job_name",
        "env",
        "env_label",
        "method",
        "seed",
        "random_repeat",
        "goal_offset",
        "eval_budget",
        "num_tasks",
        "eval_task_seed",
        "policy_seed",
        "run_label",
        "result_dir",
        "config_path",
        "dataset_file",
        "final_training_run",
    ]
    with (output_dir / "manifest.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_configs",
    )
    parser.add_argument("--job-name", default=None)
    args = parser.parse_args()

    rows = generate(args.output_dir, selected_job=args.job_name)
    write_manifest(rows, args.output_dir)
    print(f"Wrote {len(rows)} extended-offset planning configs to {args.output_dir}")


if __name__ == "__main__":
    main()
