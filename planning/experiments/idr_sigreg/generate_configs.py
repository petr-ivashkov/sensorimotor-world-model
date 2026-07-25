#!/usr/bin/env python3
"""Generate the fixed-weight IDR+SIGReg experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


METHOD = "idr_sigreg"
HISTORY = 1
INVERSE_WEIGHT = 10.0
SIGREG_WEIGHT = 0.09
SEEDS = (0, 1, 2, 3, 4)

NUM_EVAL = 100
GOAL_OFFSET_STEPS = 25
EVAL_BUDGET = 50
BASE_TASK_SEED = 42025
BASE_POLICY_SEED = 52025


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


def run_name(env: Environment, seed: int) -> str:
    return f"{env.slug}_{METHOD}_seed{seed}"


def task_seed(env_idx: int) -> int:
    return BASE_TASK_SEED + 1000 * env_idx


def policy_seed(env_idx: int, seed: int) -> int:
    return BASE_POLICY_SEED + 1000 * env_idx + seed


def train_config(env: Environment, seed: int) -> dict[str, object]:
    name = run_name(env, seed)
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
        "wandb": {"config": {"name": f"idr_sigreg_{name}"}},
        "loss": {
            "inverse": {"weight": INVERSE_WEIGHT},
            "sigreg": {"weight": SIGREG_WEIGHT},
        },
        "idr_sigreg": {
            "environment": env.label,
            "method": METHOD,
            "history": HISTORY,
            "inverse_weight": INVERSE_WEIGHT,
            "sigreg_weight": SIGREG_WEIGHT,
            "seed": seed,
        },
    }


def eval_config(
    env: Environment,
    env_idx: int,
    seed: int,
) -> dict[str, object]:
    name = run_name(env, seed)
    run_label = f"seed_{seed}"
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
                "name": run_label,
                "run_dir": "${oc.env:REPO_ROOT}/experiments/idr_sigreg/"
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
        "idr_sigreg": {
            "environment": env.label,
            "method": METHOD,
            "history": HISTORY,
            "inverse_weight": INVERSE_WEIGHT,
            "sigreg_weight": SIGREG_WEIGHT,
            "seed": seed,
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
        for seed in SEEDS:
            name = run_name(env, seed)
            train_cfg = train_config(env, seed)
            eval_cfg = eval_config(env, env_idx, seed)
            OmegaConf.save(train_cfg, train_dir / f"{name}.yaml")
            OmegaConf.save(eval_cfg, eval_dir / f"{name}.yaml")
            rows.append(
                {
                    "run_name": name,
                    "env": env.slug,
                    "env_label": env.label,
                    "method": METHOD,
                    "seed": str(seed),
                    "run_label": f"seed_{seed}",
                    "history": str(HISTORY),
                    "inverse_weight": f"{INVERSE_WEIGHT:g}",
                    "sigreg_weight": f"{SIGREG_WEIGHT:g}",
                    "train_result_dir": f"results/train/{name}",
                    "eval_result_dir": (
                        f"results/eval/{env.slug}/{METHOD}/seed_{seed}"
                    ),
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
