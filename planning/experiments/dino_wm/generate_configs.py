#!/usr/bin/env python3
"""Generate matched train and planning-eval configs for DINO-WM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


METHOD = 'dino_wm'
BACKBONE = 'facebook/dinov2-small'
BACKBONE_REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
PROTOCOL_VERSION = 'matched_batch256_mean_accumulation_v2'
HISTORY = 1
SEEDS = (0, 1, 2, 3, 4)
MAX_EPOCHS = 10
REFERENCE_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 32
GRAD_ACCUMULATION_STEPS = REFERENCE_BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 24
PREFETCH_FACTOR = 4
VALIDATION_INTERVAL_UPDATES = 500
VALIDATION_BATCHES_AT_REFERENCE_BATCH = 10
LOG_INTERVAL_UPDATES = 25

NUM_EVAL = 100
GOAL_OFFSET_STEPS = 25
EVAL_BUDGET = 50
BASE_TASK_SEED = 42025
BASE_POLICY_SEED = 52025


@dataclass(frozen=True)
class Environment:
    label: str
    slug: str
    data_config: str
    eval_config: str
    dataset_file: str
    state_key: str | None


ENVIRONMENTS = (
    Environment(
        'TwoRoom',
        'tworoom',
        'tworoom',
        'tworoom',
        'tworoom_eval.h5',
        'proprio',
    ),
    Environment(
        'Reacher',
        'reacher',
        'reacher',
        'reacher',
        'reacher_eval.h5',
        'observation',
    ),
    Environment(
        'Push-T',
        'pusht',
        'pusht',
        'pusht',
        'pusht_expert_eval.h5',
        'proprio',
    ),
    Environment(
        'OGBench-Cube',
        'ogbcube',
        'ogbcube',
        'ogbcube',
        'cube_single_expert_eval.h5',
        None,
    ),
)


def run_name(env: Environment, seed: int) -> str:
    return f'{env.slug}_{METHOD}_seed{seed}'


def task_seed(env_idx: int) -> int:
    return BASE_TASK_SEED + 1000 * env_idx


def policy_seed(env_idx: int, seed: int) -> int:
    return BASE_POLICY_SEED + 1000 * env_idx + seed


def encoding_config(env: Environment) -> dict[str, int]:
    encoding = {}
    if env.state_key is not None:
        encoding[env.state_key] = 10
    encoding['action'] = 10
    return encoding


def train_config(env: Environment, seed: int) -> dict[str, object]:
    name = run_name(env, seed)
    return {
        'defaults': [
            f'/train/data/{env.data_config}',
            {'override hydra/job_logging': 'disabled'},
            {'override hydra/hydra_logging': 'disabled'},
            '_self_',
        ],
        'hydra': {
            'searchpath': ['file://${oc.env:REPO_ROOT}/config'],
            'run': {'dir': '${oc.env:RUNS_ROOT,results/train}/${subdir}'},
            'output_subdir': None,
            'job': {'chdir': False},
        },
        'subdir': name,
        'seed': seed,
        'image_size': 224,
        'patch_size': 14,
        'num_workers': NUM_WORKERS,
        'backbone': {
            'name': BACKBONE,
            'revision': BACKBONE_REVISION,
            'type': 'dinov2_small',
            'interpolate_pos_encoding': True,
        },
        'wm': {
            'history_size': HISTORY,
            'num_preds': 1,
            'encoding': encoding_config(env),
        },
        'predictor': {
            'size': 'small',
            'depth': 6,
            'heads': 16,
            'mlp_dim': 2048,
            'dim_head': 64,
            'dropout': 0.1,
            'emb_dropout': 0.0,
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
        },
        'trainer': {
            'max_epochs': MAX_EPOCHS,
            'strategy': 'auto',
            'devices': 'auto',
            'accelerator': 'gpu',
            'precision': 'bf16',
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': GRAD_ACCUMULATION_STEPS,
            'val_check_interval': (
                VALIDATION_INTERVAL_UPDATES * GRAD_ACCUMULATION_STEPS
            ),
            'limit_val_batches': (
                VALIDATION_BATCHES_AT_REFERENCE_BATCH
                * GRAD_ACCUMULATION_STEPS
            ),
            'log_every_n_steps': (
                LOG_INTERVAL_UPDATES * GRAD_ACCUMULATION_STEPS
            ),
        },
        'loader': {
            'batch_size': MICRO_BATCH_SIZE,
            'num_workers': '${num_workers}',
            'drop_last': True,
            'persistent_workers': True,
            'prefetch_factor': PREFETCH_FACTOR,
            'pin_memory': True,
            'shuffle': True,
        },
        'data': {'dataset': {'num_steps': HISTORY + 1}},
        'optimization_matching': {
            'enabled': True,
            'protocol_version': PROTOCOL_VERSION,
            'reference_batch_size': REFERENCE_BATCH_SIZE,
            'micro_batch_size': MICRO_BATCH_SIZE,
            'accumulation_steps': GRAD_ACCUMULATION_STEPS,
            'gradient_reduction': 'mean',
            'validation_interval_updates': VALIDATION_INTERVAL_UPDATES,
            'validation_batches_at_reference_batch': (
                VALIDATION_BATCHES_AT_REFERENCE_BATCH
            ),
            'log_interval_updates': LOG_INTERVAL_UPDATES,
        },
        'artifacts': {'use_external_callbacks': False},
        'wandb': {
            'enabled': True,
            'config': {
                'project': 'sensorimotor-world-model',
                'name': f'dino_wm_{name}',
                'resume': 'never',
                'log_model': False,
            },
        },
        'dino_wm_experiment': {
            'environment': env.label,
            'method': METHOD,
            'history': HISTORY,
            'seed': seed,
            'backbone': BACKBONE,
            'backbone_revision': BACKBONE_REVISION,
            'state_key': env.state_key,
            'matched_full_training_split': True,
            'matched_max_epochs': MAX_EPOCHS,
            'matched_effective_batch_size': REFERENCE_BATCH_SIZE,
        },
    }


def eval_config(
    env: Environment,
    env_idx: int,
    seed: int,
) -> dict[str, object]:
    name = run_name(env, seed)
    run_label = f'seed_{seed}'
    return {
        'defaults': [
            '/eval/base',
            f'/eval/env/{env.eval_config}',
            {'override hydra/job_logging': 'disabled'},
            {'override hydra/hydra_logging': 'disabled'},
            '_self_',
        ],
        'hydra': {'searchpath': ['file://${oc.env:REPO_ROOT}/config']},
        'seed': policy_seed(env_idx, seed),
        'runs': [
            {
                'name': run_label,
                'run_dir': '${oc.env:REPO_ROOT}/experiments/dino_wm/'
                + f'results/train/{name}',
            }
        ],
        'eval': {
            'num_eval': NUM_EVAL,
            'goal_offset_steps': GOAL_OFFSET_STEPS,
            'eval_budget': EVAL_BUDGET,
            'task_seed': task_seed(env_idx),
            'save_video': False,
        },
        'dino_wm_experiment': {
            'environment': env.label,
            'method': METHOD,
            'history': HISTORY,
            'seed': seed,
            'task_seed': task_seed(env_idx),
            'policy_seed': policy_seed(env_idx, seed),
            'backbone': BACKBONE,
            'backbone_revision': BACKBONE_REVISION,
            'state_key': env.state_key,
        },
    }


def generate(output_dir: Path) -> list[dict[str, str]]:
    train_dir = output_dir / 'train'
    eval_dir = output_dir / 'eval'
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for env_idx, env in enumerate(ENVIRONMENTS):
        for seed in SEEDS:
            name = run_name(env, seed)
            train_cfg = train_config(env, seed)
            eval_cfg = eval_config(env, env_idx, seed)
            OmegaConf.save(train_cfg, train_dir / f'{name}.yaml')
            OmegaConf.save(eval_cfg, eval_dir / f'{name}.yaml')
            rows.append(
                {
                    'run_name': name,
                    'env': env.slug,
                    'env_label': env.label,
                    'method': METHOD,
                    'seed': str(seed),
                    'run_label': f'seed_{seed}',
                    'history': str(HISTORY),
                    'backbone': BACKBONE,
                    'backbone_revision': BACKBONE_REVISION,
                    'protocol_version': PROTOCOL_VERSION,
                    'reference_batch_size': str(REFERENCE_BATCH_SIZE),
                    'micro_batch_size': str(MICRO_BATCH_SIZE),
                    'accumulation_steps': str(GRAD_ACCUMULATION_STEPS),
                    'gradient_reduction': 'mean',
                    'state_key': env.state_key or '',
                    'train_result_dir': f'results/train/{name}',
                    'eval_result_dir': (
                        f'results/eval/{env.slug}/{METHOD}/seed_{seed}'
                    ),
                    'dataset_file': env.dataset_file,
                    'num_eval': str(NUM_EVAL),
                    'goal_offset': str(GOAL_OFFSET_STEPS),
                    'eval_budget': str(EVAL_BUDGET),
                    'eval_task_seed': str(eval_cfg['eval']['task_seed']),
                    'policy_seed': str(eval_cfg['seed']),
                }
            )
    return rows


def write_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = list(rows[0])
    with (output_dir / 'manifest.tsv').open(
        'w', newline='', encoding='utf-8'
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    queue = '\n'.join(row['run_name'] for row in rows) + '\n'
    (output_dir / 'train_queue.txt').write_text(queue, encoding='utf-8')
    (output_dir / 'eval_queue.txt').write_text(queue, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'generated_configs',
    )
    args = parser.parse_args()
    rows = generate(args.output_dir)
    write_manifest(rows, args.output_dir)
    print(
        f'Wrote {len(rows)} train + {len(rows)} eval configs '
        f'to {args.output_dir}'
    )


if __name__ == '__main__':
    main()
