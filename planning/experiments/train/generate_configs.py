#!/usr/bin/env python3
"""Generate Hydra configs for the final training sweep."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


EXPERIMENT_NAME = "train"
SIGREG_WEIGHT = 0.09
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Environment:
    label: str
    config_name: str
    run_prefix: str
    inverse_weight: float
    inverse_name_weight: float | None = None


ENVIRONMENTS = (
    Environment("TwoRoom", "tworoom", "tworoom", 0.1),
    # Keep the legacy lambda_1 run names so this targeted correction overwrites
    # the existing Reacher inverse outputs, but train them with lambda=5.
    Environment("Reacher", "reacher", "reacher", 5.0, inverse_name_weight=1.0),
    Environment("Push-T", "pusht", "pusht", 30.0),
    Environment("OGBench-Cube", "ogbcube", "cube", 1.0),
)


@dataclass(frozen=True)
class Method:
    name: str
    slug: str
    inverse_weight: float | None
    sigreg_weight: float


METHODS = (
    Method("forward-only", "forward_only", 0.0, 0.0),
    Method("inverse / ours", "inverse", None, 0.0),
    Method("sigreg", "sigreg", 0.0, SIGREG_WEIGHT),
)


def weight_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_name(env: Environment, method: Method, seed: int) -> str:
    if method.slug == "inverse":
        name_weight = (
            env.inverse_weight
            if env.inverse_name_weight is None
            else env.inverse_name_weight
        )
        return (
            f"{env.run_prefix}_inverse_lambda_"
            f"{weight_label(name_weight)}_seed{seed}"
        )
    return f"{env.run_prefix}_{method.slug}_seed{seed}"


def config_for_run(env: Environment, method: Method, seed: int) -> dict:
    inverse_weight = (
        env.inverse_weight if method.inverse_weight is None else method.inverse_weight
    )
    name = run_name(env, method, seed)
    return {
        "defaults": [
            "/train/base",
            f"/train/data/{env.config_name}",
            {"override hydra/job_logging": "disabled"},
            {"override hydra/hydra_logging": "disabled"},
            "_self_",
        ],
        "hydra": {
            "searchpath": ["file://${oc.env:REPO_ROOT}/config"],
        },
        "subdir": name,
        "seed": seed,
        "artifacts": {
            "embedding_subset_size": 4096,
        },
        "validation_monitoring": {
            "enabled": True,
        },
        "wm": {
            "history_size": 1,
        },
        "data": {
            "dataset": {
                "num_steps": 2,
            },
        },
        "wandb": {
            "config": {
                "name": f"{EXPERIMENT_NAME}_{name}",
            },
        },
        "loss": {
            "sigreg": {
                "weight": method.sigreg_weight,
            },
            "inverse": {
                "weight": inverse_weight,
            },
        },
        "final_training": {
            "environment": env.label,
            "method": method.name,
            "seed": seed,
        },
    }


def generate(output_dir: Path, selected_run: str | None = None) -> list[tuple[str, ...]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, ...]] = []
    names_seen: set[str] = set()

    for env in ENVIRONMENTS:
        for method in METHODS:
            for seed in SEEDS:
                name = run_name(env, method, seed)
                if selected_run and name != selected_run:
                    continue
                if name in names_seen:
                    raise ValueError(f"Duplicate run name generated: {name}")
                names_seen.add(name)

                cfg = OmegaConf.create(config_for_run(env, method, seed))
                OmegaConf.save(config=cfg, f=output_dir / f"{name}.yaml")

                inverse_weight = cfg.loss.inverse.weight
                sigreg_weight = cfg.loss.sigreg.weight
                rows.append(
                    (
                        name,
                        env.label,
                        method.slug,
                        str(seed),
                        f"{inverse_weight:g}",
                        f"{sigreg_weight:g}",
                        str(output_dir / f"{name}.yaml"),
                    )
                )

    if not rows:
        raise SystemExit(f"No final-training runs matched {selected_run!r}")

    return rows


def write_manifest(rows: list[tuple[str, ...]], output_dir: Path) -> None:
    header = (
        "run_name",
        "environment",
        "method",
        "seed",
        "inverse_weight",
        "sigreg_weight",
        "config_path",
    )
    manifest = output_dir / "manifest.tsv"
    manifest.write_text(
        "\n".join("\t".join(row) for row in (header, *rows)) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_configs",
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    rows = generate(args.output_dir, args.run_name)
    write_manifest(rows, args.output_dir)
    print(f"Wrote {len(rows)} generated configs to {args.output_dir}")


if __name__ == "__main__":
    main()
