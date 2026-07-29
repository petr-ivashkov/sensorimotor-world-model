#!/usr/bin/env python3
"""Aggregate planning success and held-out lambda-sweep diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DIAGNOSTIC_METRICS = (
    "effective_rank",
    "mean_per_dim_variance",
    "mean_latent_vector_length",
    "inverse_mse",
    "forward_mse",
)
SUMMARY_METRICS = ("success_rate", *DIAGNOSTIC_METRICS)
SUCCESS_ALIASES = ("success_rate", "success", "success_mean")


def parse_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), "ok"
    except json.JSONDecodeError:
        return {}, "invalid_json"


def planning_success(payload: dict[str, Any]) -> float:
    metrics = payload.get("metrics", payload)

    episode_successes = metrics.get("episode_successes")
    if isinstance(episode_successes, list):
        values = [bool(value) for value in episode_successes]
        if values:
            return 100.0 * sum(values) / len(values)

    for name in SUCCESS_ALIASES:
        if name in metrics:
            value = parse_float(metrics[name])
            if math.isfinite(value):
                return value

    values = [
        token == "True"
        for token in re.findall(r"True|False", str(episode_successes))
    ]
    return 100.0 * sum(values) / len(values) if values else math.nan


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def aggregate(exp_dir: Path) -> list[dict[str, object]]:
    rows = load_manifest(exp_dir / "generated_configs" / "manifest.tsv")
    output: list[dict[str, object]] = []

    for row in rows:
        train_dir = exp_dir / row["train_result_dir"]
        eval_dir = exp_dir / row["eval_result_dir"]
        diagnostics_path = train_dir / "diagnostics.json"
        metrics_path = eval_dir / "metrics.json"
        diagnostics, diagnostics_status = load_json(diagnostics_path)
        planning, planning_status = load_json(metrics_path)
        diagnostic_values = diagnostics.get("metrics", {})

        result: dict[str, object] = {
            "run_name": row["run_name"],
            "env": row["env"],
            "env_label": row["env_label"],
            "lambda": parse_float(row["lambda"]),
            "lambda_label": row["lambda_label"],
            "seed": int(row["seed"]),
            "history": int(row["history"]),
            "success_rate": planning_success(planning),
            "num_eval": int(row["num_eval"]),
            "goal_offset": int(row["goal_offset"]),
            "eval_budget": int(row["eval_budget"]),
            "eval_task_seed": int(row["eval_task_seed"]),
            "policy_seed": int(row["policy_seed"]),
            "diagnostics_status": diagnostics_status,
            "planning_status": planning_status,
            "diagnostics_path": str(diagnostics_path),
            "metrics_path": str(metrics_path),
        }
        for metric in DIAGNOSTIC_METRICS:
            result[metric] = parse_float(diagnostic_values.get(metric))
        result["status"] = (
            "ok"
            if diagnostics_status == "ok" and planning_status == "ok"
            else f"diagnostics_{diagnostics_status};planning_{planning_status}"
        )
        output.append(result)
    return output


def finite(values: list[object]) -> list[float]:
    parsed = [parse_float(value) for value in values]
    return [value for value in parsed if math.isfinite(value)]


def mean_std(values: list[object]) -> tuple[float, float, int]:
    values_finite = finite(values)
    n = len(values_finite)
    if not values_finite:
        return math.nan, math.nan, 0
    mean = sum(values_finite) / n
    if n == 1:
        return mean, 0.0, 1
    variance = sum((value - mean) ** 2 for value in values_finite) / (n - 1)
    return mean, math.sqrt(variance), n


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["env"]),
            str(row["env_label"]),
            float(row["lambda"]),
            str(row["lambda_label"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (env, env_label, lambda_value, lambda_label), group in groups.items():
        result: dict[str, object] = {
            "env": env,
            "env_label": env_label,
            "lambda": lambda_value,
            "lambda_label": lambda_label,
        }
        for metric in SUMMARY_METRICS:
            mean, std, n = mean_std([row[metric] for row in group])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
            result[f"{metric}_n"] = n
        summary.append(result)
    return summary


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    output = args.output or exp_dir / "aggregated_results.csv"
    summary_output = args.summary_output or exp_dir / "results" / "summary_results.csv"

    rows = aggregate(exp_dir)
    summary = summarize(rows)
    write_csv(rows, output)
    write_csv(summary, summary_output)

    diagnostics_done = sum(row["diagnostics_status"] == "ok" for row in rows)
    planning_done = sum(row["planning_status"] == "ok" for row in rows)
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Wrote {len(summary)} environment/lambda rows to {summary_output}")
    print(f"Diagnostics complete: {diagnostics_done}/{len(rows)}")
    print(f"Planning complete: {planning_done}/{len(rows)}")


if __name__ == "__main__":
    main()
