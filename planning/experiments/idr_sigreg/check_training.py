#!/usr/bin/env python3
"""Check that every IDR+SIGReg training run produced its required artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    exp_dir = args.experiment_dir.resolve()

    manifest = exp_dir / "generated_configs" / "manifest.tsv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    failures: list[str] = []
    for row in rows:
        run_dir = exp_dir / row["train_result_dir"]
        required = (
            run_dir / "config.yaml",
            run_dir / "checkpoints" / "last.ckpt",
            run_dir / "final_embeddings.pt",
        )
        missing = [
            str(path.relative_to(exp_dir)) for path in required if not path.is_file()
        ]
        if missing:
            failures.append(f"{row['run_name']}: missing {', '.join(missing)}")
            continue

        cfg = OmegaConf.load(run_dir / "config.yaml")
        observed = (
            int(cfg.wm.history_size),
            float(cfg.loss.inverse.weight),
            float(cfg.loss.sigreg.weight),
            int(cfg.seed),
        )
        expected = (
            int(row["history"]),
            float(row["inverse_weight"]),
            float(row["sigreg_weight"]),
            int(row["seed"]),
        )
        if observed != expected:
            failures.append(
                f"{row['run_name']}: config {observed} does not match manifest {expected}"
            )

    complete = len(rows) - len(failures)
    print(f"Complete and config-matched: {complete}/{len(rows)}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
