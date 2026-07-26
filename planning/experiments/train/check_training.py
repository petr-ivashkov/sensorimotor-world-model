#!/usr/bin/env python3
"""Check final-training completion and the fixed-lambda protocol."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


EXPECTED_INVERSE_WEIGHT = 10.0
EXPECTED_SIGREG_WEIGHT = 0.09


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
    if len(rows) != 60:
        failures.append(f"manifest: expected 60 rows, found {len(rows)}")

    for row in rows:
        run_name = row["run_name"]
        run_dir = exp_dir / "results" / run_name
        required = (
            run_dir / "config.yaml",
            run_dir / "checkpoints" / "last.ckpt",
            run_dir / "final_embeddings.pt",
        )
        missing = [
            str(path.relative_to(exp_dir))
            for path in required
            if not path.is_file()
        ]
        if missing:
            failures.append(f"{run_name}: missing {', '.join(missing)}")
            continue

        cfg = OmegaConf.load(run_dir / "config.yaml")
        method = row["method"]
        expected_inverse = EXPECTED_INVERSE_WEIGHT if method == "inverse" else 0.0
        expected_sigreg = EXPECTED_SIGREG_WEIGHT if method == "sigreg" else 0.0
        checks = {
            "seed": int(cfg.seed) == int(row["seed"]),
            "history": int(cfg.wm.history_size) == 1,
            "sequence length": int(cfg.data.dataset.num_steps) == 2,
            "epochs": int(cfg.trainer.max_epochs) == 10,
            "batch size": int(cfg.loader.batch_size) == 256,
            "inverse weight": float(cfg.loss.inverse.weight) == expected_inverse,
            "SIGReg weight": float(cfg.loss.sigreg.weight) == expected_sigreg,
            "manifest inverse weight": float(row["inverse_weight"])
            == expected_inverse,
            "manifest SIGReg weight": float(row["sigreg_weight"])
            == expected_sigreg,
        }
        for label, passed in checks.items():
            if not passed:
                failures.append(f"{run_name}: {label} mismatch")

    failed_runs = {failure.split(":", 1)[0] for failure in failures}
    complete = len(rows) - len(failed_runs)
    print(f"Complete and config-matched: {complete}/{len(rows)}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
