"""Generate the final seed-0 embedding export config."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent


ENV_SPECS = [
    {
        "name": "TwoRoom",
        "slug": "tworoom",
        "run_stem": "tworoom",
        "dataset_stem": "tworoom",
        "inverse_pattern": "tworoom_inverse_lambda_*_seed0",
    },
    {
        "name": "Reacher",
        "slug": "reacher",
        "run_stem": "reacher",
        "dataset_stem": "reacher",
        "inverse_pattern": "reacher_inverse_lambda_1_seed0",
    },
    {
        "name": "Push-T",
        "slug": "pusht",
        "run_stem": "pusht",
        "dataset_stem": "pusht_expert",
        "inverse_pattern": "pusht_inverse_lambda_*_seed0",
    },
    {
        "name": "OGBench-Cube",
        "slug": "ogbcube",
        "run_stem": "cube",
        "dataset_stem": "cube_single_expert",
        "inverse_pattern": "cube_inverse_lambda_*_seed0",
    },
]


METHOD_SPECS = [
    {
        "name": "inverse",
        "label": "inverse / ours",
        "pattern_key": "inverse_pattern",
    },
    {
        "name": "sigreg",
        "label": "sigreg",
        "pattern": "{run_stem}_sigreg_seed0",
    },
    {
        "name": "forward_only",
        "label": "forward-only",
        "pattern": "{run_stem}_forward_only_seed0",
    },
]


def find_one(results_dir: Path, pattern: str) -> Path:
    matches = sorted(results_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one match for {results_dir / pattern}, found {len(matches)}"
        )
    return matches[0]


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def build_config(results_dir: Path, train_suffix: str, eval_suffix: str) -> dict:
    environments = []
    for env_spec in ENV_SPECS:
        slug = env_spec["slug"]
        dataset_stem = env_spec["dataset_stem"]
        methods = []
        for method_spec in METHOD_SPECS:
            pattern = method_spec.get("pattern_key")
            if pattern is not None:
                run_pattern = env_spec[pattern]
            else:
                run_pattern = method_spec["pattern"].format(**env_spec)
            run_dir = find_one(results_dir, run_pattern)
            methods.append(
                {
                    "name": method_spec["name"],
                    "label": method_spec["label"],
                    "seed": 0,
                    "run_dir": "${oc.env:REPO_ROOT}/" + repo_relative(run_dir),
                    "output_dir": (
                        "${oc.env:REPO_ROOT}/experiments/embedding_analysis/"
                        f"outputs/{slug}/{method_spec['name']}"
                    ),
                }
            )

        environments.append(
            {
                "name": env_spec["name"],
                "slug": slug,
                "randomized_train_dataset_path": (
                    "${oc.env:EXTERNAL_DATA_ROOT,${oc.env:REPO_ROOT}/data/external}/"
                    f"{dataset_stem}_train_randomized_{train_suffix}.h5"
                ),
                "randomized_eval_dataset_path": (
                    "${oc.env:EXTERNAL_DATA_ROOT,${oc.env:REPO_ROOT}/data/external}/"
                    f"{dataset_stem}_eval_randomized_{eval_suffix}.h5"
                ),
                "methods": methods,
            }
        )

    return {
        "output_root": "${oc.env:REPO_ROOT}/experiments/embedding_analysis/outputs",
        "batch_size": 512,
        "device": "cuda",
        "overwrite": False,
        "environments": environments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "train" / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "config.yaml",
    )
    parser.add_argument("--train-suffix", default="25k")
    parser.add_argument("--eval-suffix", default="5k")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} already exists; pass --overwrite to replace it.")

    cfg = build_config(args.results_dir.resolve(), args.train_suffix, args.eval_suffix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(cfg), f=args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
