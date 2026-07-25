#!/usr/bin/env python3
"""Generate the five-seed inverse-weight sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


HISTORY = 1
SEEDS = (0, 1, 2, 3, 4)
NUM_EVAL = 100
GOAL_OFFSET_STEPS = 25
EVAL_BUDGET = 50
BASE_TASK_SEED = 42025
BASE_POLICY_SEED = 52025


@dataclass(frozen=True)
class LambdaValue:
    value: float
    label: str


LAMBDA_VALUES = (
    LambdaValue(0.0, "0"),
    LambdaValue(0.1, "0p1"),
    LambdaValue(0.3, "0p3"),
    LambdaValue(1.0, "1"),
    LambdaValue(3.0, "3"),
    LambdaValue(10.0, "10"),
    LambdaValue(30.0, "30"),
    LambdaValue(100.0, "100"),
)


@dataclass(frozen=True)
class Environment:
    label: str
    slug: str
    data_config: str
    eval_config: str
    dataset_file: str


ENVIRONMENTS = (
    Environment("TwoRoom", "tworoom", "tworoom", "tworoom", "tworoom_eval.h5"),
    Environment("Reacher", "reacher", "reacher", "reacher", "reacher_eval.h5"),
    Environment("Push-T", "pusht", "pusht", "pusht", "pusht_expert_eval.h5"),
    Environment(
        "OGBench-Cube",
        "ogbcube",
        "ogbcube",
        "ogbcube",
        "cube_single_expert_eval.h5",
    ),
)


def run_name(env: Environment, lambda_value: LambdaValue, seed: int) -> str:
    return f"{env.slug}_lambda_{lambda_value.label}_seed{seed}"


def task_seed(env_idx: int) -> int:
    return BASE_TASK_SEED + 1000 * env_idx


def policy_seed(env_idx: int, seed: int) -> int:
    return BASE_POLICY_SEED + 1000 * env_idx + seed


def train_config(
    env: Environment, lambda_value: LambdaValue, seed: int
) -> dict[str, object]:
    name = run_name(env, lambda_value, seed)
    return {
        "defaults": [
            "/train/base",
            f"/train/data/{env.data_config}",
            {"override hydra/job_logging": "disabled"},
            {"override hydra/hydra_logging": "disabled"},
            "_self_",
        ],
        "hydra": {"searchpath": ["file://${oc.env:REPO_ROOT}/config"]},
        "subdir": name,
        "seed": seed,
        "artifacts": {"embedding_subset_size": 4096},
        "validation_monitoring": {"enabled": True},
        "wm": {"history_size": HISTORY},
        "data": {"dataset": {"num_steps": HISTORY + 1}},
        "wandb": {"config": {"name": f"lambda_sweep_{name}"}},
        "loss": {
            "sigreg": {"weight": 0.0},
            "inverse": {"weight": lambda_value.value},
        },
        "lambda_sweep": {
            "environment": env.label,
            "lambda": lambda_value.value,
            "lambda_label": lambda_value.label,
            "seed": seed,
            "history": HISTORY,
        },
    }


def eval_config(
    env: Environment,
    env_idx: int,
    lambda_value: LambdaValue,
    seed: int,
) -> dict[str, object]:
    name = run_name(env, lambda_value, seed)
    return {
        "defaults": [
            "/eval/base",
            f"/eval/env/{env.eval_config}",
            {"override hydra/job_logging": "disabled"},
            {"override hydra/hydra_logging": "disabled"},
            "_self_",
        ],
        "hydra": {"searchpath": ["file://${oc.env:REPO_ROOT}/config"]},
        "seed": policy_seed(env_idx, seed),
        "runs": [
            {
                "name": name,
                "run_dir": "${oc.env:REPO_ROOT}/experiments/lambda_sweep/"
                + f"results/train/{name}",
            }
        ],
        "eval": {
            "num_eval": NUM_EVAL,
            "goal_offset_steps": GOAL_OFFSET_STEPS,
            "eval_budget": EVAL_BUDGET,
            "task_seed": task_seed(env_idx),
            "save_video": False,
        },
        "lambda_sweep": {
            "environment": env.label,
            "lambda": lambda_value.value,
            "lambda_label": lambda_value.label,
            "seed": seed,
            "history": HISTORY,
            "task_seed": task_seed(env_idx),
            "policy_seed": policy_seed(env_idx, seed),
        },
    }


def generate(output_dir: Path) -> list[dict[str, str]]:
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for env_idx, env in enumerate(ENVIRONMENTS):
        for lambda_value in LAMBDA_VALUES:
            for seed in SEEDS:
                name = run_name(env, lambda_value, seed)
                train_cfg = train_config(env, lambda_value, seed)
                eval_cfg = eval_config(env, env_idx, lambda_value, seed)
                OmegaConf.save(train_cfg, train_dir / f"{name}.yaml")
                OmegaConf.save(eval_cfg, eval_dir / f"{name}.yaml")
                rows.append(
                    {
                        "run_name": name,
                        "env": env.slug,
                        "env_label": env.label,
                        "lambda": f"{lambda_value.value:g}",
                        "lambda_label": lambda_value.label,
                        "seed": str(seed),
                        "history": str(HISTORY),
                        "train_result_dir": f"results/train/{name}",
                        "eval_result_dir": f"results/eval/{name}",
                        "dataset_file": env.dataset_file,
                        "num_eval": str(NUM_EVAL),
                        "goal_offset": str(GOAL_OFFSET_STEPS),
                        "eval_budget": str(EVAL_BUDGET),
                        "eval_task_seed": str(eval_cfg["eval"]["task_seed"]),
                        "policy_seed": str(eval_cfg["seed"]),
                    }
                )
    return rows


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = list(rows[0])
    with (output_dir / "manifest.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    names = [row["run_name"] for row in rows]
    queue = "\n".join(names) + "\n"
    (output_dir / "train_queue.txt").write_text(queue, encoding="utf-8")
    (output_dir / "eval_queue.txt").write_text(queue, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_configs",
    )
    args = parser.parse_args()
    rows = generate(args.output_dir)
    write_manifest(rows, args.output_dir)
    print(f"Wrote {len(rows)} train + {len(rows)} eval configs to {args.output_dir}")


if __name__ == "__main__":
    main()
