#!/usr/bin/env python3
"""Generate one-run planning-eval configs for the final trained models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


EXPERIMENT_NAME = "planning_eval"
TRAINING_EXPERIMENT = "train"
SEEDS = (0, 1, 2, 3, 4)
GOAL_OFFSET_STEPS = 25
EVAL_BUDGET = 50
NUM_EVAL = 100
BASE_TASK_SEED = 42025
BASE_POLICY_SEED = 52025


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


def task_seed(env_idx: int) -> int:
    # One fixed task set per environment for all methods, seeds, and repeats.
    return BASE_TASK_SEED + 1000 * env_idx


def policy_seed(env_idx: int, seed_or_repeat: int) -> int:
    # Keep random repeats independent without changing the sampled tasks.
    return BASE_POLICY_SEED + 1000 * env_idx + seed_or_repeat


def config_for_job(
    env: Environment,
    env_idx: int,
    method: str,
    seed_or_repeat: int,
    training_runs: dict[tuple[str, str, int], str],
) -> tuple[str, dict]:
    is_random = method == "random"
    run_label = f"repeat_{seed_or_repeat}" if is_random else f"seed_{seed_or_repeat}"
    job_name = f"{env.slug}_{method}_{run_label}"

    run: dict[str, object] = {"name": run_label}
    final_training_run = None
    if is_random:
        run["policy"] = "random"
    else:
        final_training_run = training_runs[(env.label, method, seed_or_repeat)]
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
        "seed": policy_seed(env_idx, seed_or_repeat),
        "runs": [run],
        "eval": {
            "num_eval": NUM_EVAL,
            "goal_offset_steps": GOAL_OFFSET_STEPS,
            "eval_budget": EVAL_BUDGET,
            "task_seed": task_seed(env_idx),
            "save_video": False,
        },
        "planning_eval": {
            "environment": env.label,
            "env_slug": env.slug,
            "method": method,
            "seed_or_repeat": seed_or_repeat,
            "run_label": run_label,
            "result_dir": f"results/{env.slug}/{method}/{run_label}",
            "final_training_run": final_training_run,
            "num_eval": NUM_EVAL,
            "goal_offset_steps": GOAL_OFFSET_STEPS,
            "eval_budget": EVAL_BUDGET,
            "task_seed": task_seed(env_idx),
            "policy_seed": policy_seed(env_idx, seed_or_repeat),
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
        for method in METHODS:
            for seed_or_repeat in SEEDS:
                job_name, cfg = config_for_job(
                    env, env_idx, method, seed_or_repeat, training_runs
                )
                if selected_job and job_name != selected_job:
                    continue

                config_path = output_dir / f"{job_name}.yaml"
                OmegaConf.save(config=OmegaConf.create(cfg), f=config_path)

                run_label = cfg["planning_eval"]["run_label"]
                rows.append(
                    {
                        "job_name": job_name,
                        "env": env.slug,
                        "env_label": env.label,
                        "method": method,
                        "seed_or_repeat": str(seed_or_repeat),
                        "run_label": str(run_label),
                        "eval_task_seed": str(cfg["eval"]["task_seed"]),
                        "policy_seed": str(cfg["seed"]),
                        "num_eval": str(NUM_EVAL),
                        "goal_offset": str(GOAL_OFFSET_STEPS),
                        "eval_budget": str(EVAL_BUDGET),
                        "result_dir": f"results/{env.slug}/{method}/{run_label}",
                        "config_path": str(config_path),
                        "dataset_file": env.dataset_file,
                        "final_training_run": str(
                            cfg["planning_eval"]["final_training_run"] or ""
                        ),
                    }
                )

    if not rows:
        raise SystemExit(f"No planning-eval jobs matched {selected_job!r}")

    return rows


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = [
        "job_name",
        "env",
        "env_label",
        "method",
        "seed_or_repeat",
        "run_label",
        "eval_task_seed",
        "policy_seed",
        "num_eval",
        "goal_offset",
        "eval_budget",
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
    print(f"Wrote {len(rows)} generated planning-eval configs to {args.output_dir}")


if __name__ == "__main__":
    main()
