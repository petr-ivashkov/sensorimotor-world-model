"""Episode-level deterministic train/eval split for HDF5 datasets.

Splits episodes (not clips) so that planning evaluation is held out at the
trajectory level. Seed is fixed at 3072, ratio is 90/10.

Outputs (siblings of the source files):
    pusht_expert.h5        ->  pusht_expert_train.h5, pusht_expert_eval.h5
    tworoom.h5             ->  tworoom_train.h5, tworoom_eval.h5
    cube_single_expert.h5  ->  cube_single_expert_train.h5,
                               cube_single_expert_eval.h5
    reacher.h5             ->  reacher_train.h5, reacher_eval.h5

The pre-existing ``pusht_expert_train.h5`` is the *full* pusht dataset and
must be renamed to ``pusht_expert.h5`` before running this script.

For each output file, ``ep_len``, ``ep_offset`` and the per-row episode-id
column (``episode_idx`` or ``ep_idx``) are rewritten so the file is
self-consistent. All other per-row and per-episode columns are sliced and
copied as-is.
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

SEED = 3072
TRAIN_FRAC = 0.9
PER_ROW_EP_ID_CANDIDATES = ("episode_idx", "ep_idx")
# h5py only accepts these strings as the `compression` create_dataset arg.
# Sources written with third-party filters report `compression == "unknown"`
# (or another unrecognized name) and must be re-emitted uncompressed.
WRITABLE_COMPRESSIONS = frozenset({"gzip", "lzf", "szip"})


def output_filter_kwargs(src_d: h5py.Dataset) -> dict:
    comp = src_d.compression
    if comp in WRITABLE_COMPRESSIONS:
        return {"compression": comp, "compression_opts": src_d.compression_opts}
    if comp is None:
        return {}
    # Source uses a filter h5py won't accept as a write arg (e.g. blosc via
    # hdf5plugin). Prefer Blosc/Zstd through hdf5plugin (much faster than
    # gzip with comparable ratio); fall back to gzip if hdf5plugin is absent.
    if hdf5plugin is not None:
        return dict(hdf5plugin.Blosc(cname="zstd", clevel=3,
                                     shuffle=hdf5plugin.Blosc.SHUFFLE))
    return {"compression": "gzip", "compression_opts": 4}


def fit_chunks(src_d: h5py.Dataset, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if src_d.chunks is None:
        return None
    return tuple(min(c, s) if s > 0 else 1 for c, s in zip(src_d.chunks, shape))


@dataclass(frozen=True)
class SplitJob:
    source: str
    train_out: str
    eval_out: str


JOBS: tuple[SplitJob, ...] = (
    SplitJob("pusht_expert.h5", "pusht_expert_train.h5", "pusht_expert_eval.h5"),
    SplitJob("tworoom.h5", "tworoom_train.h5", "tworoom_eval.h5"),
    SplitJob(
        "cube_single_expert.h5",
        "cube_single_expert_train.h5",
        "cube_single_expert_eval.h5",
    ),
    SplitJob("reacher.h5", "reacher_train.h5", "reacher_eval.h5"),
)


def detect_ep_id_column(f: h5py.File, total_rows: int) -> str:
    for name in PER_ROW_EP_ID_CANDIDATES:
        if name in f and f[name].shape[0] == total_rows:
            return name
    raise KeyError(
        f"no per-row episode-id column among {PER_ROW_EP_ID_CANDIDATES} "
        f"(total rows = {total_rows})"
    )


def split_episode_indices(num_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(num_episodes)
    n_train = int(round(TRAIN_FRAC * num_episodes))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def write_split(
    src_path: Path,
    dst_path: Path,
    selected: np.ndarray,
    ep_id_col: str,
) -> None:
    with h5py.File(src_path, "r") as src:
        ep_len = src["ep_len"][:]
        ep_offset = src["ep_offset"][:]
        num_episodes = len(ep_len)
        total_rows = int(ep_len.sum())

        new_ep_len = ep_len[selected].astype(ep_len.dtype, copy=True)
        new_total = int(new_ep_len.sum())
        new_ep_offset = np.empty(len(selected), dtype=ep_offset.dtype)
        if len(selected):
            new_ep_offset[0] = 0
            np.cumsum(new_ep_len[:-1], out=new_ep_offset[1:])

        per_row: list[str] = []
        per_episode: list[str] = []
        unknown: list[str] = []
        for name in src:
            obj = src[name]
            if not isinstance(obj, h5py.Dataset) or name in ("ep_len", "ep_offset"):
                continue
            n = obj.shape[0] if obj.shape else None
            if n == total_rows:
                per_row.append(name)
            elif n == num_episodes:
                per_episode.append(name)
            else:
                unknown.append(f"{name} (shape={obj.shape})")
        if unknown:
            raise RuntimeError(
                f"{src_path}: cannot classify datasets {unknown} "
                f"(total_rows={total_rows}, num_episodes={num_episodes})"
            )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst_path.with_name(dst_path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        with h5py.File(tmp_path, "w") as dst:
            dst.create_dataset("ep_len", data=new_ep_len)
            dst.create_dataset("ep_offset", data=new_ep_offset)

            new_ep_id_values = np.repeat(
                np.arange(len(selected)), new_ep_len.astype(np.int64)
            )

            for name in per_row:
                src_d = src[name]
                shape = (new_total,) + tuple(src_d.shape[1:])
                filt = output_filter_kwargs(src_d)
                dst_d = dst.create_dataset(
                    name,
                    shape=shape,
                    dtype=src_d.dtype,
                    chunks=fit_chunks(src_d, shape) if filt else None,
                    **filt,
                )
                for k, v in src_d.attrs.items():
                    dst_d.attrs[k] = v

                if name == ep_id_col:
                    dst_d[:] = new_ep_id_values.astype(src_d.dtype, copy=False)
                    continue

                dst_off = 0
                for src_ep in selected:
                    L = int(ep_len[src_ep])
                    if L == 0:
                        continue
                    s = int(ep_offset[src_ep])
                    dst_d[dst_off:dst_off + L] = src_d[s:s + L]
                    dst_off += L

            for name in per_episode:
                src_d = src[name]
                shape = (len(selected),) + tuple(src_d.shape[1:])
                filt = output_filter_kwargs(src_d)
                dst_d = dst.create_dataset(
                    name,
                    shape=shape,
                    dtype=src_d.dtype,
                    chunks=fit_chunks(src_d, shape) if filt else None,
                    **filt,
                )
                for k, v in src_d.attrs.items():
                    dst_d.attrs[k] = v
                dst_d[:] = src_d[:][selected]

            for k, v in src.attrs.items():
                dst.attrs[k] = v

        tmp_path.rename(dst_path)


def run_job(root: Path, job: SplitJob, force: bool) -> None:
    src_path = root / job.source
    if not src_path.exists():
        print(f"[skip] {src_path} missing")
        return

    with h5py.File(src_path, "r") as f:
        ep_len = f["ep_len"][:]
        total_rows = int(ep_len.sum())
        ep_id_col = detect_ep_id_column(f, total_rows)

    train_eps, eval_eps = split_episode_indices(len(ep_len))
    print(
        f"[{job.source}] episodes={len(ep_len)} "
        f"train={len(train_eps)} eval={len(eval_eps)} ep_id={ep_id_col}"
    )

    for out_name, eps in ((job.train_out, train_eps), (job.eval_out, eval_eps)):
        out = root / out_name
        if out.exists() and not force:
            print(f"  exists  {out}")
            continue
        print(f"  writing {out}")
        write_split(src_path, out, eps, ep_id_col)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=os.environ.get("EXTERNAL_DATA_ROOT", "data/external"),
        help="Directory containing the source HDF5 files.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split files.",
    )
    args = ap.parse_args()
    root = Path(args.root)
    for job in JOBS:
        run_job(root, job, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
