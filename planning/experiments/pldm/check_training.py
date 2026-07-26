#!/usr/bin/env python3
"""Check PLDM training completion and matched-compute configuration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


EXPECTED_LOSSES = {
    'sigreg': (True, 0.0),
    'temp_straight': (False, 0.1),
    'std': (True, 18.0),
    'std_t': (True, 0.7),
    'cov': (True, 12.0),
    'cov_t': (True, 0.0),
    'temp_align': (True, 0.2),
    'idm': (True, 0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    exp_dir = args.experiment_dir.resolve()

    manifest = exp_dir / 'generated_configs' / 'manifest.tsv'
    with manifest.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))

    failures: list[str] = []
    for row in rows:
        run_dir = exp_dir / row['train_result_dir']
        required = (
            run_dir / 'config.yaml',
            run_dir / 'checkpoints' / 'last.ckpt',
            run_dir / 'final_embeddings.pt',
        )
        missing = [
            str(path.relative_to(exp_dir))
            for path in required
            if not path.is_file()
        ]
        if missing:
            failures.append(
                f"{row['run_name']}: missing {', '.join(missing)}"
            )
            continue

        cfg = OmegaConf.load(run_dir / 'config.yaml')
        if int(cfg.wm.history_size) != 1:
            failures.append(f"{row['run_name']}: history is not 1")
        updates_per_epoch = int(
            cfg.pldm_experiment.get('optimizer_updates_per_epoch', 0)
        )
        matched_checks = {
            'epochs': int(cfg.trainer.max_epochs) == 10,
            'batch size': int(cfg.loader.batch_size) == 256,
            'learning rate': float(cfg.optimizer.lr) == 1e-4,
            'weight decay': float(cfg.optimizer.weight_decay) == 1e-3,
            'precision': str(cfg.trainer.precision) == 'bf16',
            'optimizer updates': updates_per_epoch > 0,
        }
        for label, passed in matched_checks.items():
            if not passed:
                failures.append(f"{row['run_name']}: {label} mismatch")
        if int(cfg.seed) != int(row['seed']):
            failures.append(f"{row['run_name']}: seed mismatch")

        observed_losses = {
            key: (bool(cfg.loss[key].enabled), float(cfg.loss[key].weight))
            for key in EXPECTED_LOSSES
        }
        if observed_losses != EXPECTED_LOSSES:
            failures.append(f"{row['run_name']}: loss config mismatch")

    complete = len(rows) - len({failure.split(':', 1)[0] for failure in failures})
    print(f'Complete and config-matched: {complete}/{len(rows)}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
