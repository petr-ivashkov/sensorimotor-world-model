"""Create small randomized transition HDF5 datasets for analysis.

The full canonical datasets live under ``data/external``. This script samples valid
one-step transitions ``(o_t, a_t, o_{t+1})`` without loading full pixel arrays into memory.

Usage:

    python scripts/create_randomized_transition_datasets.py \
      --data-dir data/external \
      --train-n 100000 \
      --eval-n 10000 \
      --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin  # registers Blosc/Zstd/etc. plugins on import
except ImportError:
    hdf5plugin = None


EP_ID_KEYS = ("episode_idx", "ep_idx")
ACTION_KEYS = ("action", "actions")
WRITABLE_COMPRESSIONS = frozenset({"gzip", "lzf", "szip"})


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    stem: str
    aliases: tuple[str, ...] = ()


SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("TwoRoom", "tworoom"),
    DatasetSpec("Reacher", "reacher"),
    DatasetSpec("Push-T", "pusht_expert", ("pusht",)),
    DatasetSpec("OGBench-Cube", "cube_single_expert", ("ogbcube", "cube")),
)


def output_filter_kwargs(src_d: h5py.Dataset) -> dict:
    comp = src_d.compression
    if comp in WRITABLE_COMPRESSIONS:
        return {"compression": comp, "compression_opts": src_d.compression_opts}
    if comp is None:
        return {}
    if hdf5plugin is not None:
        return dict(
            hdf5plugin.Blosc(
                cname="zstd",
                clevel=3,
                shuffle=hdf5plugin.Blosc.SHUFFLE,
            )
        )
    return {"compression": "gzip", "compression_opts": 4}


def fit_chunks(src_d: h5py.Dataset, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if src_d.chunks is None:
        return None
    return tuple(min(c, s) if s > 0 else 1 for c, s in zip(src_d.chunks, shape))


def copy_attrs(src: h5py.Dataset | h5py.File, dst: h5py.Dataset | h5py.File) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def find_existing_key(f: h5py.File, candidates: tuple[str, ...]) -> str | None:
    return next((key for key in candidates if key in f), None)


def infer_total_rows(f: h5py.File) -> int:
    if "ep_len" in f:
        return int(np.asarray(f["ep_len"][:], dtype=np.int64).sum())
    sizes = [
        obj.shape[0]
        for obj in f.values()
        if isinstance(obj, h5py.Dataset) and obj.shape
    ]
    if not sizes:
        raise ValueError("could not infer row count from an empty HDF5 file")
    values, counts = np.unique(np.asarray(sizes, dtype=np.int64), return_counts=True)
    return int(values[np.argmax(counts)])


def find_source_file(data_dir: Path, spec: DatasetSpec, split: str) -> Path | None:
    preferred = data_dir / f"{spec.stem}_{split}.h5"
    if preferred.exists():
        return preferred

    stems = (spec.stem,) + spec.aliases
    candidates: list[Path] = []
    for stem in stems:
        candidates.extend(data_dir.glob(f"*{stem}*{split}*.h5"))
    candidates = [
        path
        for path in candidates
        if "randomized" not in path.name and path.is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def valid_transition_rows(f: h5py.File, total_rows: int) -> np.ndarray:
    if "ep_len" in f and "ep_offset" in f:
        ep_len = np.asarray(f["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(f["ep_offset"][:], dtype=np.int64)
        pieces = [
            np.arange(start, start + length - 1, dtype=np.int64)
            for start, length in zip(ep_offset, ep_len)
            if length > 1
        ]
        if pieces:
            return np.concatenate(pieces)
        return np.empty(0, dtype=np.int64)

    ep_key = find_existing_key(f, EP_ID_KEYS)
    if ep_key is None:
        return np.arange(0, max(total_rows - 1, 0), dtype=np.int64)

    ep_id = np.asarray(f[ep_key][:], dtype=np.int64)
    return np.nonzero(ep_id[:-1] == ep_id[1:])[0].astype(np.int64, copy=False)


def classify_datasets(
    f: h5py.File,
    total_rows: int,
) -> tuple[list[str], list[str], int | None]:
    num_episodes = len(f["ep_len"]) if "ep_len" in f else None
    row_names: list[str] = []
    episode_names: list[str] = []
    for name, obj in f.items():
        if not isinstance(obj, h5py.Dataset) or not obj.shape:
            continue
        if name in ("ep_len", "ep_offset"):
            continue
        if obj.shape[0] == total_rows:
            row_names.append(name)
        elif num_episodes is not None and obj.shape[0] == num_episodes:
            episode_names.append(name)
    return sorted(row_names), sorted(episode_names), num_episodes


def episode_indices_for_rows(
    f: h5py.File,
    rows: np.ndarray,
) -> np.ndarray | None:
    if "ep_offset" in f:
        ep_offset = np.asarray(f["ep_offset"][:], dtype=np.int64)
        return np.searchsorted(ep_offset, rows, side="right") - 1

    ep_key = find_existing_key(f, EP_ID_KEYS)
    if ep_key is not None:
        return np.asarray(f[ep_key][rows], dtype=np.int64)

    return None


def create_output_dataset(
    dst: h5py.File,
    name: str,
    src_d: h5py.Dataset,
    sample_count: int,
) -> h5py.Dataset:
    shape = (sample_count,) + tuple(src_d.shape[1:])
    filt = output_filter_kwargs(src_d)
    dst_d = dst.create_dataset(
        name,
        shape=shape,
        dtype=src_d.dtype,
        chunks=fit_chunks(src_d, shape) if filt else None,
        **filt,
    )
    copy_attrs(src_d, dst_d)
    return dst_d


def copy_indexed_rows(
    src_d: h5py.Dataset,
    dst_d: h5py.Dataset,
    rows: np.ndarray,
    batch_size: int,
) -> None:
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_rows = rows[start:stop]
        unique_rows, inverse = np.unique(batch_rows, return_inverse=True)
        dst_d[start:stop] = src_d[unique_rows][inverse]


def write_randomized_dataset(
    src_path: Path,
    dst_path: Path,
    sample_count: int,
    rng: np.random.Generator,
    overwrite: bool,
    batch_size: int,
) -> dict:
    if dst_path.exists() and not overwrite:
        return {"status": "exists", "path": str(dst_path)}

    with h5py.File(src_path, "r") as src:
        total_rows = infer_total_rows(src)
        action_key = find_existing_key(src, ACTION_KEYS)
        if action_key is None:
            raise KeyError(f"{src_path}: no action column among {ACTION_KEYS}")

        row_datasets, episode_datasets, _num_episodes = classify_datasets(
            src,
            total_rows,
        )
        if action_key not in row_datasets:
            raise ValueError(f"{src_path}: action column {action_key!r} is not per-row")

        valid_rows = valid_transition_rows(src, total_rows)
        if len(valid_rows) == 0:
            raise ValueError(f"{src_path}: no valid one-step transitions found")

        replace = len(valid_rows) < sample_count
        sampled_rows = rng.choice(valid_rows, size=sample_count, replace=replace)
        next_rows = sampled_rows + 1
        sampled_episodes = episode_indices_for_rows(src, sampled_rows)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_name(dst_path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        with h5py.File(tmp_path, "w") as dst:
            copy_attrs(src, dst)
            dst.attrs["source_file"] = str(src_path)
            dst.attrs["sample_count"] = sample_count
            dst.attrs["sampled_with_replacement"] = replace
            dst.attrs["transition_horizon"] = 1
            dst.attrs["action_key"] = action_key

            dst.create_dataset("source_row_idx", data=sampled_rows)
            dst.create_dataset("source_next_row_idx", data=next_rows)
            if sampled_episodes is not None:
                dst.create_dataset("source_episode_idx", data=sampled_episodes)

            for name in row_datasets:
                dst_d = create_output_dataset(dst, name, src[name], sample_count)
                copy_indexed_rows(src[name], dst_d, sampled_rows, batch_size)

                if name not in ACTION_KEYS:
                    next_name = f"{name}_next"
                    dst_next = create_output_dataset(
                        dst,
                        next_name,
                        src[name],
                        sample_count,
                    )
                    copy_indexed_rows(src[name], dst_next, next_rows, batch_size)

            if sampled_episodes is not None:
                for name in episode_datasets:
                    dst_d = create_output_dataset(dst, name, src[name], sample_count)
                    copy_indexed_rows(src[name], dst_d, sampled_episodes, batch_size)

        tmp_path.rename(dst_path)

    return {
        "status": "created",
        "path": str(dst_path),
        "source": str(src_path),
        "rows": sample_count,
        "valid_transitions": int(len(valid_rows)),
        "replace": replace,
        "columns": len(row_datasets),
        "episode_columns": len(episode_datasets),
        "action_key": action_key,
    }


def randomized_name(src_path: Path, sample_count: int) -> str:
    if sample_count % 1000 == 0:
        suffix = f"{sample_count // 1000}k"
    else:
        suffix = str(sample_count)
    return f"{src_path.stem}_randomized_{suffix}.h5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("EXTERNAL_DATA_ROOT", "data/external"),
        help="Directory containing full train/eval HDF5 datasets.",
    )
    parser.add_argument("--train-n", type=int, default=100_000)
    parser.add_argument("--eval-n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Rows copied per HDF5 read batch. Lower this to reduce pixel RAM.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)
    summaries: list[dict] = []

    for spec in SPECS:
        for split, sample_count in (("train", args.train_n), ("eval", args.eval_n)):
            src_path = find_source_file(data_dir, spec, split)
            if src_path is None:
                print(f"[skip] {spec.label} {split}: no matching HDF5 file in {data_dir}")
                continue
            dst_path = data_dir / randomized_name(src_path, sample_count)
            print(f"[{spec.label} {split}] {src_path.name} -> {dst_path.name}")
            summaries.append(
                write_randomized_dataset(
                    src_path=src_path,
                    dst_path=dst_path,
                    sample_count=sample_count,
                    rng=rng,
                    overwrite=args.overwrite,
                    batch_size=args.batch_size,
                )
            )

    print("\nSummary")
    for item in summaries:
        if item["status"] == "exists":
            print(f"  exists  {item['path']}")
            continue
        print(
            "  created "
            f"{item['path']} rows={item['rows']} "
            f"valid_source_transitions={item['valid_transitions']} "
            f"replace={item['replace']} "
            f"row_columns={item['columns']} "
            f"episode_columns={item['episode_columns']} "
            f"action={item['action_key']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
