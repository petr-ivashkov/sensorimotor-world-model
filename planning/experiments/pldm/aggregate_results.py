#!/usr/bin/env python3
"""Aggregate PLDM planning metrics and summarize them by environment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    'success_rate': ('success_rate', 'success', 'success_mean'),
    'distance_to_goal': (
        'distance_to_goal',
        'final_distance',
        'distance',
        'mean_distance',
        'goal_distance',
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
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def aggregate(exp_dir: Path) -> list[dict[str, str]]:
    rows = load_manifest(exp_dir / 'generated_configs' / 'manifest.tsv')
    output: list[dict[str, str]] = []
    for row in rows:
        run_dir = exp_dir / row['eval_result_dir']
        metrics_path = run_dir / 'metrics.json'
        status = 'missing'
        result: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            try:
                result = json.loads(metrics_path.read_text(encoding='utf-8'))
                metrics = result.get('metrics', {})
                status = 'ok'
            except json.JSONDecodeError:
                status = 'invalid_json'

        output.append(
            {
                'run_name': row['run_name'],
                'env': row['env'],
                'env_label': row['env_label'],
                'method': row['method'],
                'seed': row['seed'],
                'run_label': row['run_label'],
                'history': row['history'],
                'source_commit': row['source_commit'],
                'success_rate': str(
                    pick_metric(metrics, METRIC_ALIASES['success_rate'])
                ),
                'distance_to_goal': str(
                    pick_metric(metrics, METRIC_ALIASES['distance_to_goal'])
                ),
                'num_eval': row['num_eval'],
                'goal_offset': row['goal_offset'],
                'eval_budget': row['eval_budget'],
                'eval_task_seed': row['eval_task_seed'],
                'policy_seed': row['policy_seed'],
                'elapsed_seconds': str(
                    parse_float(result.get('elapsed_seconds'))
                ),
                'status': status,
                'run_dir': str(run_dir),
                'metrics_path': str(metrics_path),
            }
        )
    return output


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = parse_float(row['success_rate'])
        if row['status'] == 'ok' and math.isfinite(value):
            key = (row['env'], row['env_label'], row['method'])
            grouped.setdefault(key, []).append(value)

    summary: list[dict[str, str]] = []
    for (env, env_label, method), values in grouped.items():
        sem = (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else math.nan
        )
        summary.append(
            {
                'env': env,
                'env_label': env_label,
                'method': method,
                'mean': str(statistics.mean(values)),
                'sem': str(sem),
                'n': str(len(values)),
            }
        )
    return summary


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    output = args.output or exp_dir / 'aggregated_results.csv'
    rows = aggregate(exp_dir)
    write_csv(rows, output)

    summary = summarize(rows)
    if summary:
        summary_path = exp_dir / 'pldm_summary_results.csv'
        write_csv(summary, summary_path)
        print(f'Wrote {len(summary)} rows to {summary_path}')

    done = sum(row['status'] == 'ok' for row in rows)
    print(f'Wrote {len(rows)} rows to {output}')
    print(f'Completed metrics: {done}/{len(rows)}')


if __name__ == '__main__':
    main()
