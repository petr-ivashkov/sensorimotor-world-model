#!/usr/bin/env python3
"""Check DINO-WM completion and matched/default configuration."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from omegaconf import OmegaConf


EXPECTED_PROTOCOL = 'matched_batch256_mean_accumulation_v2'
EXPECTED_EPOCHS = 10
EXPECTED_REFERENCE_BATCH = 256
EXPECTED_MICRO_BATCH = 32
EXPECTED_ACCUMULATION = 8
EXPECTED_NUM_WORKERS = 24
EXPECTED_PREFETCH_FACTOR = 4
EXPECTED_VALIDATION_INTERVAL = 500 * EXPECTED_ACCUMULATION
EXPECTED_VALIDATION_BATCHES = 10 * EXPECTED_ACCUMULATION
EXPECTED_LOG_INTERVAL = 25 * EXPECTED_ACCUMULATION


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
    if len(rows) != 20:
        failures.append(f'manifest: expected 20 rows, found {len(rows)}')
    for row in rows:
        run_dir = exp_dir / row['train_result_dir']
        checkpoint_path = run_dir / 'checkpoints' / 'last.ckpt'
        required = (
            run_dir / 'config.yaml',
            checkpoint_path,
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
        accumulation = int(cfg.trainer.get('accumulate_grad_batches', 1))
        updates_per_epoch = int(
            cfg.dino_wm_experiment.get('optimizer_updates_per_epoch', 0)
        )
        expected_total_updates = updates_per_epoch * EXPECTED_EPOCHS
        total_updates = int(
            cfg.dino_wm_experiment.get('total_optimizer_updates', 0)
        )
        train_batch_limit = int(cfg.trainer.get('limit_train_batches', -1))
        matching = cfg.get('optimization_matching', {})
        checks = {
            'history': int(cfg.wm.history_size) == 1,
            'sequence length': int(cfg.data.dataset.num_steps) == 2,
            'epochs': int(cfg.trainer.max_epochs) == EXPECTED_EPOCHS,
            'batch size': int(cfg.loader.batch_size)
            == EXPECTED_MICRO_BATCH,
            'gradient accumulation': accumulation
            == EXPECTED_ACCUMULATION,
            'effective batch size': (
                int(cfg.loader.batch_size) * accumulation
                == EXPECTED_REFERENCE_BATCH
            ),
            'learning rate': float(cfg.optimizer.lr) == 1e-4,
            'weight decay': float(cfg.optimizer.weight_decay) == 1e-3,
            'precision': str(cfg.trainer.precision) == 'bf16',
            'gradient clipping': float(cfg.trainer.gradient_clip_val) == 1.0,
            'strategy': str(cfg.trainer.strategy) == 'auto',
            'workers': int(cfg.loader.num_workers)
            == EXPECTED_NUM_WORKERS,
            'prefetch factor': int(cfg.loader.prefetch_factor)
            == EXPECTED_PREFETCH_FACTOR,
            'validation interval': int(cfg.trainer.val_check_interval)
            == EXPECTED_VALIDATION_INTERVAL,
            'validation batches': int(cfg.trainer.limit_val_batches)
            == EXPECTED_VALIDATION_BATCHES,
            'logging interval': int(cfg.trainer.log_every_n_steps)
            == EXPECTED_LOG_INTERVAL,
            'matching enabled': bool(matching.get('enabled', False)),
            'protocol version': str(
                matching.get('protocol_version', '')
            ) == EXPECTED_PROTOCOL,
            'reference batch size': int(
                matching.get('reference_batch_size', 0)
            ) == EXPECTED_REFERENCE_BATCH,
            'declared micro-batch size': int(
                matching.get('micro_batch_size', 0)
            ) == EXPECTED_MICRO_BATCH,
            'declared accumulation': int(
                matching.get('accumulation_steps', 0)
            ) == EXPECTED_ACCUMULATION,
            'mean gradient reduction': str(
                matching.get('gradient_reduction', '')
            ) == 'mean',
            'declared validation interval': int(
                matching.get('validation_interval_updates', 0)
            ) == EXPECTED_VALIDATION_INTERVAL // EXPECTED_ACCUMULATION,
            'declared validation batches': int(
                matching.get('validation_batches_at_reference_batch', 0)
            ) == EXPECTED_VALIDATION_BATCHES // EXPECTED_ACCUMULATION,
            'declared logging interval': int(
                matching.get('log_interval_updates', 0)
            ) == EXPECTED_LOG_INTERVAL // EXPECTED_ACCUMULATION,
            'optimizer updates': updates_per_epoch > 0,
            'exact update limit': train_batch_limit
            == accumulation * updates_per_epoch,
            'micro-batches per epoch': int(
                cfg.dino_wm_experiment.get('micro_batches_per_epoch', 0)
            ) == accumulation * updates_per_epoch,
            'total optimizer updates': total_updates
            == expected_total_updates,
            'examples per epoch': int(
                cfg.dino_wm_experiment.get('examples_per_epoch', 0)
            ) == updates_per_epoch * EXPECTED_REFERENCE_BATCH,
            'scheduler max steps': int(
                cfg.dino_wm_experiment.get('scheduler_max_steps', 0)
            ) == expected_total_updates,
            'scheduler warmup steps': int(
                cfg.dino_wm_experiment.get(
                    'scheduler_warmup_steps',
                    0,
                )
            ) == max(1, int(0.01 * expected_total_updates)),
            'seed': int(cfg.seed) == int(row['seed']),
            'backbone': str(cfg.backbone.name) == row['backbone'],
            'backbone revision': str(cfg.backbone.revision)
            == row['backbone_revision'],
            'manifest protocol': row.get('protocol_version', '')
            == EXPECTED_PROTOCOL,
            'manifest reference batch': int(
                row.get('reference_batch_size', 0)
            ) == EXPECTED_REFERENCE_BATCH,
            'manifest micro-batch': int(
                row.get('micro_batch_size', 0)
            ) == EXPECTED_MICRO_BATCH,
            'manifest accumulation': int(
                row.get('accumulation_steps', 0)
            ) == EXPECTED_ACCUMULATION,
            'manifest gradient reduction': row.get(
                'gradient_reduction',
                '',
            ) == 'mean',
            'encoding': dict(cfg.wm.encoding)
            == expected_encoding(row['state_key']),
        }
        for label, passed in checks.items():
            if not passed:
                failures.append(f"{row['run_name']}: {label} mismatch")

        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
            mmap=True,
        )
        if int(checkpoint.get('global_step', -1)) != expected_total_updates:
            failures.append(
                f"{row['run_name']}: checkpoint optimizer-step mismatch"
            )
        scheduler_states = checkpoint.get('lr_schedulers', [])
        if len(scheduler_states) != 1:
            failures.append(
                f"{row['run_name']}: expected one scheduler state"
            )
        else:
            scheduler_state = scheduler_states[0]
            if int(scheduler_state.get('max_steps', -1)) != expected_total_updates:
                failures.append(
                    f"{row['run_name']}: checkpoint scheduler length mismatch"
                )
            expected_warmup = max(1, int(0.01 * expected_total_updates))
            if int(scheduler_state.get('warmup_steps', -1)) != expected_warmup:
                failures.append(
                    f"{row['run_name']}: checkpoint scheduler warmup mismatch"
                )
        if len(checkpoint.get('optimizer_states', [])) != 1:
            failures.append(
                f"{row['run_name']}: expected one optimizer state"
            )
        del checkpoint

    failed_runs = {failure.split(':', 1)[0] for failure in failures}
    print(f'Complete and config-matched: {len(rows) - len(failed_runs)}/{len(rows)}')
    if failures:
        print('\n'.join(failures))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
