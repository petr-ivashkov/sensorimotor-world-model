"""Export embeddings for seed-0 final trained models on randomized datasets."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

try:
    import hdf5plugin  # registers Blosc/Zstd/etc. HDF5 filters when needed
except ImportError:
    hdf5plugin = None


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))


PIXEL_KEYS = {"pixels", "pixels_next", "image", "images", "observation_pixels"}


def safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser().resolve()


def h5_len(path: Path) -> int:
    with h5py.File(path, "r") as f:
        for key in ("source_row_idx", "pixels", "action", "actions"):
            if key in f:
                return int(f[key].shape[0])
        lengths = [
            v.shape[0]
            for v in f.values()
            if isinstance(v, h5py.Dataset) and v.shape
        ]
    if not lengths:
        raise ValueError(f"{path} has no row datasets")
    return int(max(lengths))


def h5_keys(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return sorted(k for k, v in f.items() if isinstance(v, h5py.Dataset))


def ensure_nchw(pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels)
    if pixels.ndim == 3:
        pixels = pixels[None]
    if pixels.ndim != 4:
        raise ValueError(f"expected NHWC or NCHW pixels, got shape={pixels.shape}")
    if pixels.shape[-1] in (1, 3, 4):
        pixels = np.moveaxis(pixels, -1, 1)
    return np.ascontiguousarray(pixels)


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    dataset_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    n = h5_len(dataset_path)
    chunks: list[torch.Tensor] = []
    model.eval()

    with h5py.File(dataset_path, "r") as f:
        if "pixels" not in f:
            raise KeyError(f"{dataset_path} has no 'pixels' dataset")
        for start in tqdm(
            range(0, n, batch_size),
            desc=f"encode {dataset_path.name}",
            leave=False,
        ):
            stop = min(start + batch_size, n)
            pixels_np = ensure_nchw(f["pixels"][start:stop])
            pixels = torch.from_numpy(pixels_np).to(device, non_blocking=True)
            out = model.encode({"pixels": pixels})
            chunks.append(out["emb"][:, 0].detach().cpu().float())

    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)


def should_save_side_array(name: str, ds: h5py.Dataset, n_rows: int) -> bool:
    if name in PIXEL_KEYS or name.startswith("pixels"):
        return False
    if not ds.shape or ds.shape[0] != n_rows:
        return False
    return np.issubdtype(ds.dtype, np.number) or np.issubdtype(ds.dtype, np.bool_)


def read_side_arrays(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    arrays: dict[str, torch.Tensor] = {}
    manifest: dict[str, Any] = {}
    n = h5_len(path)
    with h5py.File(path, "r") as f:
        for name, ds in f.items():
            if not isinstance(ds, h5py.Dataset) or not should_save_side_array(name, ds, n):
                continue
            arr = ds[:]
            if np.issubdtype(arr.dtype, np.floating):
                tensor = torch.from_numpy(arr).float()
            elif np.issubdtype(arr.dtype, np.bool_):
                tensor = torch.from_numpy(arr.astype(np.bool_))
            else:
                tensor = torch.from_numpy(arr)
            arrays[name] = tensor
            manifest[name] = {
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
            }
    return arrays, manifest


def select_environments(config: dict[str, Any], selected: str | None) -> list[dict[str, Any]]:
    envs = list(config.get("environments", []))
    if selected is None:
        return envs
    selected_norm = safe_name(selected)
    return [
        env
        for env in envs
        if safe_name(str(env.get("name", ""))) == selected_norm
        or safe_name(str(env.get("slug", ""))) == selected_norm
    ]


def select_methods(env: dict[str, Any], selected: str | None) -> list[dict[str, Any]]:
    methods = list(env.get("methods", []))
    if selected is None:
        return methods
    return [method for method in methods if str(method.get("name")) == selected]


def verify_config(config: dict[str, Any], selected_env: str | None, selected_method: str | None) -> int:
    missing = 0
    for env in select_environments(config, selected_env):
        train_path = as_path(env["randomized_train_dataset_path"])
        eval_path = as_path(env["randomized_eval_dataset_path"])
        for split, path in (("train", train_path), ("eval", eval_path)):
            ok = path.exists()
            print(f"[{env['slug']}] {split} dataset: {path} {'OK' if ok else 'MISSING'}")
            missing += 0 if ok else 1
        for method in select_methods(env, selected_method):
            run_dir = as_path(method["run_dir"])
            ckpt = run_dir / "checkpoints" / "last.ckpt"
            cfg = run_dir / "config.yaml"
            ok = ckpt.exists() and cfg.exists()
            print(
                f"[{env['slug']}/{method['name']}] run: {run_dir} "
                f"{'OK' if ok else 'MISSING'}"
            )
            missing += 0 if ok else 1
    return missing


def save_split_payload(
    path: Path,
    *,
    embeddings: torch.Tensor,
    side_arrays: dict[str, torch.Tensor],
    side_manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    payload = {
        "embeddings": embeddings,
        "arrays": side_arrays,
        "array_manifest": side_manifest,
        "metadata": metadata,
    }
    torch.save(payload, tmp)
    tmp.rename(path)


def export_one(
    *,
    env: dict[str, Any],
    method: dict[str, Any],
    output_root: Path,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    env_name = str(env["name"])
    env_slug = str(env["slug"])
    method_name = str(method["name"])
    run_dir = as_path(method["run_dir"])
    train_path = as_path(env["randomized_train_dataset_path"])
    eval_path = as_path(env["randomized_eval_dataset_path"])
    out_dir = as_path(method.get("output_dir", output_root / env_slug / method_name))
    train_out = out_dir / "train_embeddings.pt"
    eval_out = out_dir / "eval_embeddings.pt"

    if train_out.exists() and eval_out.exists() and not overwrite:
        print(f"[exists] {env_slug}/{method_name}: {out_dir}")
        return {"env": env_slug, "method": method_name, "status": "exists", "output_dir": str(out_dir)}

    for path in (train_path, eval_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not (run_dir / "config.yaml").exists() or not (run_dir / "checkpoints" / "last.ckpt").exists():
        raise FileNotFoundError(f"missing config/checkpoint under {run_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {env_name} / {method_name} ===")
    print(f"run:   {run_dir}")
    print(f"train: {train_path} ({h5_len(train_path):,} rows)")
    print(f"eval:  {eval_path} ({h5_len(eval_path):,} rows)")

    from eval import load_jepa_from_run

    t0 = time.perf_counter()
    model, train_cfg = load_jepa_from_run(run_dir, device=str(device))
    load_seconds = time.perf_counter() - t0

    split_summaries: dict[str, Any] = {}
    for split, dataset_path, output_path in (
        ("train", train_path, train_out),
        ("eval", eval_path, eval_out),
    ):
        t_split = time.perf_counter()
        embeddings = encode_pixels(model, dataset_path, device=device, batch_size=batch_size)
        encode_seconds = time.perf_counter() - t_split
        side_arrays, side_manifest = read_side_arrays(dataset_path)

        metadata = {
            "env": env_name,
            "env_slug": env_slug,
            "method": method_name,
            "method_label": method.get("label", method_name),
            "seed": method.get("seed"),
            "split": split,
            "run_dir": str(run_dir),
            "dataset_path": str(dataset_path),
            "dataset_keys": h5_keys(dataset_path),
            "embedding_shape": list(embeddings.shape),
            "batch_size": batch_size,
            "device": str(device),
            "encode_seconds": encode_seconds,
        }
        save_split_payload(
            output_path,
            embeddings=embeddings,
            side_arrays=side_arrays,
            side_manifest=side_manifest,
            metadata=metadata,
        )
        split_summaries[split] = metadata
        print(f"[{split}] wrote {output_path} in {encode_seconds:.1f}s")

    summary = {
        "env": env_name,
        "env_slug": env_slug,
        "method": method_name,
        "method_label": method.get("label", method_name),
        "seed": method.get("seed"),
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "load_model_seconds": load_seconds,
        "total_seconds": time.perf_counter() - t0,
        "splits": split_summaries,
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    OmegaConf.save(config=OmegaConf.create(resolved_config), f=out_dir / "export_config.yaml")
    OmegaConf.save(config=train_cfg, f=out_dir / "model_train_config.yaml")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[done] {env_slug}/{method_name}: {summary['total_seconds']:.1f}s")
    return {"env": env_slug, "method": method_name, "status": "created", "output_dir": str(out_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--env", help="Optional environment name/slug.")
    parser.add_argument("--method", help="Optional method name: inverse, sigreg, forward_only.")
    parser.add_argument("--output-root", help="Override config output_root.")
    parser.add_argument("--batch-size", type=int, help="Override config batch_size.")
    parser.add_argument("--device", help="Override config device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exports.")
    parser.add_argument("--verify-only", action="store_true", help="Only verify configured paths.")
    args = parser.parse_args()

    cfg_node = OmegaConf.load(args.config)
    resolved = OmegaConf.to_container(cfg_node, resolve=True)
    assert isinstance(resolved, dict)

    if args.verify_only:
        missing = verify_config(resolved, args.env, args.method)
        print(f"Verification complete: {missing} missing path(s).")
        return 1 if missing else 0

    output_root = as_path(args.output_root or resolved["output_root"])
    batch_size = int(args.batch_size or resolved.get("batch_size", 512))
    device_name = str(args.device or resolved.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    overwrite = bool(args.overwrite or resolved.get("overwrite", False))

    envs = select_environments(resolved, args.env)
    if not envs:
        raise SystemExit(f"No environments matched --env={args.env!r}")

    results = []
    for env in envs:
        methods = select_methods(env, args.method)
        if not methods:
            print(f"[skip] {env.get('slug')}: no methods matched --method={args.method!r}")
            continue
        for method in methods:
            results.append(
                export_one(
                    env=env,
                    method=method,
                    output_root=output_root,
                    device=device,
                    batch_size=batch_size,
                    overwrite=overwrite,
                    resolved_config=resolved,
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "last_export_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
