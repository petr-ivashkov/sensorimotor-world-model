#!/usr/bin/env python3
"""Check DINO-WM completion and matched/default configuration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


def expected_encoding(state_key: str) -> dict[str, int]:
    encoding = {}
    if state_key:
        encoding[state_key] = 10
    encoding['action'] = 10
    return encoding


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
        checks = {
            'history': int(cfg.wm.history_size) == 1,
            'sequence length': int(cfg.data.dataset.num_steps) == 2,
            'epochs': int(cfg.trainer.max_epochs) == 10,
            'batch size': int(cfg.loader.batch_size) == 32,
            'learning rate': float(cfg.optimizer.lr) == 5e-4,
            'weight decay': float(cfg.optimizer.weight_decay) == 0.0,
            'seed': int(cfg.seed) == int(row['seed']),
            'backbone': str(cfg.backbone.name) == row['backbone'],
            'backbone revision': str(cfg.backbone.revision)
            == row['backbone_revision'],
            'encoding': dict(cfg.wm.encoding)
            == expected_encoding(row['state_key']),
        }
        for label, passed in checks.items():
            if not passed:
                failures.append(f"{row['run_name']}: {label} mismatch")

    failed_runs = {failure.split(':', 1)[0] for failure in failures}
    print(f'Complete and config-matched: {len(rows) - len(failed_runs)}/{len(rows)}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
