"""Plot measurements exported by the online pilot; no model or cluster access.

Usage: python -m planning.experiments.online_adaptation.plot_results OUTPUT_ROOT
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_rounds(output_root: str | Path) -> list[dict]:
    """Read real round metrics and order them by the recorded round number."""
    paths = sorted((Path(output_root) / "metrics").glob("round_*.json"))
    if not paths:
        raise FileNotFoundError(f"No metrics/round_*.json files under {output_root}")
    records = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        round_number = record.get("round")
        transitions = record.get("experience_transitions")
        if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 0:
            raise ValueError(f"{path}: round must be a nonnegative integer")
        if round_number in seen:
            raise ValueError(f"{path}: duplicate round {round_number}")
        if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
            raise ValueError(f"{path}: experience_transitions must be a nonnegative integer")
        seen.add(round_number)
        records.append(record)
    return sorted(records, key=lambda record: record["round"])


def _series(records, domain, section, metric):
    x, y = [], []
    for record in records:
        result = record.get(domain) or {}
        values = result.get(section) or {}
        value = values.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            x.append(record["experience_transitions"])
            y.append(value)
    return x, y


def make_figure(records: list[dict]):
    """Build six headless panels; skip unavailable measurements, including CEM.

    Success uses the pinned environment adapter's ``success_rate`` percentage.
    Error panels use the physical-unit diagnostics, avoiding comparisons of
    latent MSE across encoder snapshots. No smoothing or interpolation is used.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    if not records:
        raise ValueError("No round records to plot")
    figure = Figure(figsize=(12, 6.7), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3).ravel()
    domains = (("target", "Target", "#2066a8", "-"),
               ("source", "Source", "#d47264", "--"))
    panels = (
        ("Fixed-anchor representation", "Linear CKA", "diagnostics",
         (("representation.linear_cka", "", None),)),
        ("Representation coordinate change", "Relative coordinate drift", "diagnostics",
         (("representation.relative_coordinate_drift", "", None),)),
        ("Five-step block prediction", "Position RMSE (pixels)", "diagnostics",
         (("horizon_5.prediction.block_position_rmse", "prediction", None),
          ("horizon_5.persistence.block_position_rmse", "persistence", ":"))),
        ("Observed-embedding block probe", "Position RMSE (pixels)", "diagnostics",
         (("observed.block_position_rmse", "", None),)),
        ("Fixed-task CEM control", "Success (%)", "planning",
         (("success_rate", "", None),)),
        ("Five-step orientation prediction", "Circular angle MAE (radians)", "diagnostics",
         (("horizon_5.prediction.block_angle_mae_rad", "prediction", None),
          ("horizon_5.persistence.block_angle_mae_rad", "persistence", ":"))),
    )
    for axis, (title, ylabel, section, curves) in zip(axes, panels):
        for domain, label, color, domain_style in domains:
            for metric, suffix, style in curves:
                x, y = _series(records, domain, section, metric)
                if x:
                    axis.plot(x, y, label=f"{label} {suffix}".strip(), color=color,
                              linestyle=style or domain_style, marker="o", markersize=3,
                              linewidth=1.4, alpha=0.75 if suffix == "persistence" else 1)
        axis.set(title=title, xlabel="New training transitions", ylabel=ylabel)
        axis.grid(alpha=0.18)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        if axis.lines:
            axis.legend(fontsize=8, frameon=False)
        else:
            axis.text(0.5, 0.5, "No recorded measurements", ha="center", va="center",
                      transform=axis.transAxes, color="0.45", fontsize=9)
    axes[0].set_ylim(-0.02, 1.02)
    axes[4].set_ylim(-2, 102)
    figure.suptitle("Online adaptation measurements", fontsize=14)
    return figure


def plot_results(output_root: str | Path, output: str | Path | None = None) -> Path:
    output_root = Path(output_root)
    destination = Path(output) if output is not None else output_root / "adaptation_curves.pdf"
    records = load_rounds(output_root)
    figure = make_figure(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    figure.clear()
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(plot_results(args.output_root, args.output))


if __name__ == "__main__":
    main()
