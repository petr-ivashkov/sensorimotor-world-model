#!/usr/bin/env python3
"""Generate train + eval configs for the history-length ablation.

Grid: 4 environments x {IDR, SIGReg} x history in {1,2,3} x 5 seeds = 120 runs.
IDR uses a fixed inverse weight lambda=10 on every run; SIGReg uses weight 0.09.
Everything else is inherited from /train/base, /train/data/<env>, /eval/base and
/eval/env/<env>, so no training or evaluation logic is duplicated here.

Writes, under --output-dir (default generated_configs/):
  train/<run_name>.yaml   eval/<run_name>.yaml
  manifest.tsv            train_queue.txt   eval_queue.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


LAMBDA_IDR = 10.0
SIGREG_WEIGHT = 0.09
HISTORIES = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)

# Reuse planning_eval's task/policy seed scheme so the ablation evaluates on the
# same 100 tasks per environment as the main Fig. 5 comparison.
BASE_TASK_SEED = 42025
BASE_POLICY_SEED = 52025
NUM_EVAL = 100
GOAL_OFFSET_STEPS = 25
EVAL_BUDGET = 50

RESULTS_TRAIN = "results/train"
RESULTS_EVAL = "results/eval"


@dataclass(frozen=True)
class Environment:
    label: str
    slug: str
    data_config: str   # /train/data/<data_config>
    eval_config: str    # /eval/env/<eval_config>
    dataset_file: str


ENVIRONMENTS = (
    Environment("TwoRoom", "tworoom", "tworoom", "tworoom", "tworoom_eval.h5"),
    Environment("Reacher", "reacher", "reacher", "reacher", "reacher_eval.h5"),
    Environment("Push-T", "pusht", "pusht", "pusht", "pusht_expert_eval.h5"),
    Environment(
        "OGBench-Cube", "ogbcube", "ogbcube", "ogbcube", "cube_single_expert_eval.h5"
    ),
)


@dataclass(frozen=True)
class Method:
    name: str
    inverse_weight: float
    sigreg_weight: float


METHODS = (
    Method("idr", LAMBDA_IDR, 0.0),
    Method("sigreg", 0.0, SIGREG_WEIGHT),
)


def run_name(env: Environment, method: Method, history: int, seed: int) -> str:
    return f"{env.slug}_{method.name}_h{history}_seed{seed}"


def train_config(env: Environment, method: Method, history: int, seed: int) -> dict:
    name = run_name(env, method, history, seed)
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
        "wm": {"history_size": history},
        "data": {"dataset": {"num_steps": history + 1}},
        "wandb": {"config": {"name": f"history_ablation_{name}"}},
        "loss": {
            "sigreg": {"weight": method.sigreg_weight},
            "inverse": {"weight": method.inverse_weight},
        },
        "history_ablation": {
            "environment": env.label,
            "method": method.name,
            "history": history,
            "seed": seed,
        },
    }


def eval_config(
    env: Environment, env_idx: int, method: Method, history: int, seed: int
) -> dict:
    name = run_name(env, method, history, seed)
    task_seed = BASE_TASK_SEED + 1000 * env_idx
    policy_seed = BASE_POLICY_SEED + 1000 * env_idx + seed
    return {
        "defaults": [
            "/eval/base",
            f"/eval/env/{env.eval_config}",
            {"override hydra/job_logging": "disabled"},
            {"override hydra/hydra_logging": "disabled"},
            "_self_",
        ],
        "hydra": {"searchpath": ["file://${oc.env:REPO_ROOT}/config"]},
        "seed": policy_seed,
        "runs": [
            {
                "name": name,
                "run_dir": "${oc.env:REPO_ROOT}/experiments/history_ablation/"
                + f"{RESULTS_TRAIN}/{name}",
            }
        ],
        "eval": {
            "num_eval": NUM_EVAL,
            "goal_offset_steps": GOAL_OFFSET_STEPS,
            "eval_budget": EVAL_BUDGET,
            "task_seed": task_seed,
            "save_video": False,
        },
        "history_ablation": {
            "environment": env.label,
            "method": method.name,
            "history": history,
            "seed": seed,
            "task_seed": task_seed,
            "policy_seed": policy_seed,
        },
    }


def generate(output_dir: Path) -> list[dict[str, str]]:
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for env_idx, env in enumerate(ENVIRONMENTS):
        for method in METHODS:
            for history in HISTORIES:
                for seed in SEEDS:
                    name = run_name(env, method, history, seed)
                    tcfg = train_config(env, method, history, seed)
                    ecfg = eval_config(env, env_idx, method, history, seed)
                    OmegaConf.save(OmegaConf.create(tcfg), train_dir / f"{name}.yaml")
                    OmegaConf.save(OmegaConf.create(ecfg), eval_dir / f"{name}.yaml")
                    rows.append(
                        {
                            "run_name": name,
                            "env": env.slug,
                            "env_label": env.label,
                            "method": method.name,
                            "history": str(history),
                            "seed": str(seed),
                            "inverse_weight": f"{method.inverse_weight:g}",
                            "sigreg_weight": f"{method.sigreg_weight:g}",
                            "result_dir": f"{RESULTS_EVAL}/{name}",
                            "dataset_file": env.dataset_file,
                            "num_eval": str(NUM_EVAL),
                            "goal_offset": str(GOAL_OFFSET_STEPS),
                            "eval_budget": str(EVAL_BUDGET),
                            "eval_task_seed": str(ecfg["eval"]["task_seed"]),
                            "policy_seed": str(ecfg["seed"]),
                        }
                    )
    return rows


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    header = list(rows[0].keys())
    lines = ["\t".join(header)]
    lines += ["\t".join(row[k] for k in header) for row in rows]
    (output_dir / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    names = [row["run_name"] for row in rows]
    (output_dir / "train_queue.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output_dir / "eval_queue.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


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
