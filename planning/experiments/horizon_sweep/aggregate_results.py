#!/usr/bin/env python3
"""Aggregate extended-offset planning metrics into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "success_rate": ("success_rate", "success", "success_mean"),
    "distance_to_goal": (
        "distance_to_goal",
        "final_distance",
        "distance",
        "mean_distance",
        "goal_distance",
    ),
}


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def pick_metric(metrics: dict[str, Any], aliases: tuple[str, ...]) -> float:
    for key in aliases:
        if key in metrics:
            return parse_float(metrics[key])
    return math.nan


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def aggregate(exp_dir: Path) -> list[dict[str, str]]:
    manifest_path = exp_dir / "generated_configs" / "manifest.tsv"
    rows = load_manifest(manifest_path)

    out_rows: list[dict[str, str]] = []
    for row in rows:
        result_dir = exp_dir / row["result_dir"]
        metrics_path = result_dir / "metrics.json"
        status = "missing"
        result: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            try:
                result = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = result.get("metrics", {})
                status = "ok"
            except json.JSONDecodeError:
                status = "invalid_json"

        out_rows.append(
            {
                "env": row["env"],
                "env_label": row["env_label"],
                "method": row["method"],
                "seed": row["seed"],
                "random_repeat": row["random_repeat"],
                "goal_offset": row["goal_offset"],
                "eval_budget": row["eval_budget"],
                "num_tasks": row["num_tasks"],
                "success_rate": str(
                    pick_metric(metrics, METRIC_ALIASES["success_rate"])
                ),
                "distance_to_goal": str(
                    pick_metric(metrics, METRIC_ALIASES["distance_to_goal"])
                ),
                "eval_task_seed": row["eval_task_seed"],
                "policy_seed": row["policy_seed"],
                "elapsed_seconds": str(parse_float(result.get("elapsed_seconds"))),
                "status": status,
                "result_dir": str(result_dir),
                "metrics_path": str(metrics_path),
                "final_training_run": row["final_training_run"],
            }
        )
    return out_rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "env",
        "env_label",
        "method",
        "seed",
        "random_repeat",
        "goal_offset",
        "eval_budget",
        "num_tasks",
        "success_rate",
        "distance_to_goal",
        "eval_task_seed",
        "policy_seed",
        "elapsed_seconds",
        "status",
        "result_dir",
        "metrics_path",
        "final_training_run",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <experiment-dir>/aggregated_results.csv",
    )
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    output = args.output or exp_dir / "aggregated_results.csv"
    rows = aggregate(exp_dir)
    write_csv(rows, output)

    done = sum(row["status"] == "ok" for row in rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Completed metrics: {done}/{len(rows)}")


if __name__ == "__main__":
    main()
