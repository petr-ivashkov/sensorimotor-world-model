"""Read planning evaluations and summarize per-run success in percent."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = [
    "env", "environment", "method", "inverse_weight", "sigreg_weight",
    "history", "goal_offset", "eval_budget", "num_eval", "task_seed",
]
COLUMNS = [*SETTINGS, "train_seed", "repeat", "policy_seed", "success_rate", "path"]


def load_results(root):
    rows = []
    for path in sorted(Path(root).rglob("metrics.json")):
        result = json.loads(path.read_text())
        outcomes = np.asarray(result["metrics"]["episode_successes"], dtype=float)
        if outcomes.ndim != 1 or len(outcomes) != result["num_eval"]:
            raise ValueError(f"Wrong episode count in {path}")
        if not len(outcomes) or not np.isin(outcomes, [0, 1]).all():
            raise ValueError(f"Invalid episode outcomes in {path}")
        success = 100 * outcomes.mean()
        if not np.isclose(success, float(result["metrics"]["success_rate"])):
            raise ValueError(f"Reported success differs from episode outcomes in {path}")
        row = {key: result[key] for key in SETTINGS}
        row.update(
            train_seed=result["train_seed"], repeat=result["repeat"],
            policy_seed=result["seed"], success_rate=success, path=str(path),
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(runs):
    """Equal weight per training seed (policy repeat for Random); sample STD."""
    if runs.empty:
        return pd.DataFrame(columns=[*SETTINGS, "mean", "std", "n"])
    duplicate = runs.duplicated([*SETTINGS, "train_seed", "repeat"], keep=False)
    if duplicate.any():
        raise ValueError("Repeated seed/configuration. Select one evaluation per training run or policy repeat.")
    return runs.groupby(SETTINGS, dropna=False, as_index=False).agg(
        mean=("success_rate", "mean"), std=("success_rate", "std"), n=("success_rate", "size"),
    )
