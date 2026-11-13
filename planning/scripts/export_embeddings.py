"""Export frozen train/eval embeddings and physical quantities for one model."""

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

try:
    import hdf5plugin
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("REPO_ROOT", str(REPO_ROOT))
PIXEL_KEYS = {"pixels", "pixels_next", "image", "images", "observation_pixels"}


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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--train-data", type=Path, help="Explicit training-image HDF5 file.")
    parser.add_argument("--eval-data", type=Path, help="Explicit evaluation-image HDF5 file.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/embeddings")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (args.train_data is None) != (args.eval_data is None):
        parser.error("Pass both --train-data and --eval-data.")
    run_dir = args.run_dir.expanduser().resolve()
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if cfg.method in {"pldm", "dino_wm"}:
        raise ValueError("These representation/probe analyses use SMWM, SIGReg and Forward-only embeddings.")
    data_dir = args.data_dir or Path(cfg.data.dataset.cache_dir)
    stem = str(cfg.data.dataset.name).removesuffix("_train")
    inputs = {
        "train": data_dir / f"{stem}_train_randomized_25k.h5",
        "eval": data_dir / f"{stem}_eval_randomized_5k.h5",
    }
    if args.train_data is not None:
        inputs = {"train": args.train_data, "eval": args.eval_data}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"{path}; run scripts/create_randomized_transition_datasets.py first.")
    output = args.output_dir / cfg.env / cfg.method
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Embedding output exists: {output}. Choose another --output-dir.")
    from eval import load_jepa_from_run

    model, _ = load_jepa_from_run(run_dir, device=args.device)
    output.mkdir(parents=True, exist_ok=True)
    for split, path in inputs.items():
        embeddings = encode_pixels(model, path, device=torch.device(args.device), batch_size=args.batch_size)
        arrays, manifest = read_side_arrays(path)
        target = output / f"{split}_embeddings.pt"
        save_split_payload(
            target, embeddings=embeddings, side_arrays=arrays, side_manifest=manifest,
            metadata={"run_dir": str(run_dir), "env": cfg.env, "method": cfg.method,
                      "seed": int(cfg.seed), "split": split, "dataset": str(path)},
        )
        print(f"Saved {target}: {tuple(embeddings.shape)}")


if __name__ == "__main__":
    main()
