#!/usr/bin/env python3
"""Generate matched train and planning-eval configs for PLDM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


METHOD = 'pldm'
SOURCE_COMMIT = '0baacc8118c5f262aeff942da75cf191e71d3c5f'
HISTORY = 1
SEEDS = (0, 1, 2, 3, 4)
MAX_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3

NUM_EVAL = 100
GOAL_OFFSET_STEPS = 25
EVAL_BUDGET = 50
BASE_TASK_SEED = 42025
BASE_POLICY_SEED = 52025

LOSS_WEIGHTS = {
    'sigreg': (True, 0.0),
    'temp_straight': (False, 0.1),
    'std': (True, 18.0),
    'std_t': (True, 0.7),
    'cov': (True, 12.0),
    'cov_t': (True, 0.0),
    'temp_align': (True, 0.2),
    'idm': (True, 0.0),
}


@dataclass(frozen=True)
class Environment:
    label: str
    slug: str
    data_config: str
    eval_config: str
    dataset_file: str


ENVIRONMENTS = (
    Environment('TwoRoom', 'tworoom', 'tworoom', 'tworoom', 'tworoom_eval.h5'),
    Environment('Reacher', 'reacher', 'reacher', 'reacher', 'reacher_eval.h5'),
    Environment('Push-T', 'pusht', 'pusht', 'pusht', 'pusht_expert_eval.h5'),
    Environment(
        'OGBench-Cube',
        'ogbcube',
        'ogbcube',
        'ogbcube',
        'cube_single_expert_eval.h5',
    ),
)


def run_name(env: Environment, seed: int) -> str:
    return f'{env.slug}_{METHOD}_seed{seed}'


def task_seed(env_idx: int) -> int:
    return BASE_TASK_SEED + 1000 * env_idx


def policy_seed(env_idx: int, seed: int) -> int:
    return BASE_POLICY_SEED + 1000 * env_idx + seed


def loss_config() -> dict[str, dict[str, object]]:
    return {
        name: {'enabled': enabled, 'weight': weight}
        for name, (enabled, weight) in LOSS_WEIGHTS.items()
    }


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
        'img_size': 224,
        'patch_size': 14,
        'encoder_scale': 'tiny',
        'num_workers': 6,
        'trainer': {
            'max_epochs': MAX_EPOCHS,
            'devices': 'auto',
            'accelerator': 'gpu',
            'precision': 'bf16',
            'gradient_clip_val': 1.0,
        },
        'loader': {
            'batch_size': BATCH_SIZE,
            'num_workers': '${num_workers}',
            'drop_last': True,
            'persistent_workers': True,
            'prefetch_factor': 3,
            'pin_memory': True,
            'shuffle': True,
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
        },
        'wm': {
            'history_size': HISTORY,
            'num_preds': 1,
            'embed_dim': 192,
            'use_proprio': False,
        },
        'data': {'dataset': {'num_steps': HISTORY + 1}},
        'model': {
            '_target_': 'vendor.pldm.pldm.PLDM',
            'encoder': {
                '_target_': 'stable_pretraining.backbone.utils.vit_hf',
                'size': '${encoder_scale}',
                'patch_size': '${patch_size}',
                'image_size': '${img_size}',
                'pretrained': False,
                'use_mask_token': False,
            },
            'predictor': {
                '_target_': 'vendor.pldm.module.Predictor',
                'num_frames': '${wm.history_size}',
                'input_dim': '${wm.embed_dim}',
                'hidden_dim': '${wm.embed_dim}',
                'output_dim': '${wm.embed_dim}',
                'depth': 6,
                'heads': 16,
                'mlp_dim': 2048,
                'dim_head': 64,
                'dropout': 0.1,
                'emb_dropout': 0.0,
            },
            'action_encoder': {
                '_target_': 'vendor.pldm.module.Embedder',
                'input_dim': '???',
                'emb_dim': '${wm.embed_dim}',
            },
            'projector': {
                '_target_': 'vendor.pldm.module.MLP',
                'input_dim': '${wm.embed_dim}',
                'output_dim': '${wm.embed_dim}',
                'hidden_dim': 2048,
                'norm_fn': {
                    '_target_': 'torch.nn.BatchNorm1d',
                    '_partial_': True,
                },
            },
            'pred_proj': {
                '_target_': 'vendor.pldm.module.MLP',
                'input_dim': '${wm.embed_dim}',
                'output_dim': '${wm.embed_dim}',
                'hidden_dim': 2048,
                'norm_fn': {
                    '_target_': 'torch.nn.BatchNorm1d',
                    '_partial_': True,
                },
            },
        },
        'idm': {
            '_target_': 'vendor.pldm.module.MLP',
            'input_dim': '???',
            'hidden_dim': 512,
            'output_dim': '???',
        },
        'loss': loss_config(),
        'artifacts': {
            'embedding_subset_size': 4096,
            'use_external_callbacks': False,
        },
        'wandb': {
            'enabled': True,
            'config': {
                'project': 'sensorimotor-world-model',
                'name': f'pldm_{name}',
                'resume': 'never',
                'log_model': False,
            },
        },
        'pldm_experiment': {
            'environment': env.label,
            'method': METHOD,
            'history': HISTORY,
            'seed': seed,
            'source_commit': SOURCE_COMMIT,
            'matched_full_training_split': True,
            'matched_max_epochs': MAX_EPOCHS,
            'matched_effective_batch_size': BATCH_SIZE,
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
                'run_dir': '${oc.env:REPO_ROOT}/experiments/pldm/'
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
        'pldm_experiment': {
            'environment': env.label,
            'method': METHOD,
            'history': HISTORY,
            'seed': seed,
            'task_seed': task_seed(env_idx),
            'policy_seed': policy_seed(env_idx, seed),
            'source_commit': SOURCE_COMMIT,
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
                    'source_commit': SOURCE_COMMIT,
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

    names = [row['run_name'] for row in rows]
    queue = '\n'.join(names) + '\n'
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
    print(f'Wrote {len(rows)} train + {len(rows)} eval configs to {args.output_dir}')


if __name__ == '__main__':
    main()
