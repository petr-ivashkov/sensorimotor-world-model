"""Export small inspection artifacts without copying model weights or datasets."""
import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401: register SWM dataset compression
import imageio.v2 as imageio
import numpy as np

from planning.experiments.online_adaptation.plot_results import load_rounds, plot_results


def export_review(output_root):
    root = Path(output_root)
    paths = sorted((root / "metrics").glob("round_*.json"))
    if paths:
        plot_results(root)
        records = load_rounds(root)
        if "observed.block2_position_rmse" in records[-1]["target"]["diagnostics"]:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            figure = Figure(figsize=(10, 4), constrained_layout=True)
            FigureCanvasAgg(figure)
            axes = figure.subplots(1, 2)
            for axis, mode in zip(axes, ("observed", "horizon_5.prediction")):
                for block in ("block", "block2"):
                    axis.plot([r["experience_transitions"] for r in records],
                              [r["target"]["diagnostics"][f"{mode}.{block}_position_rmse"] for r in records],
                              marker="o", label=block)
                axis.set(title=mode, xlabel="New training transitions", ylabel="Position RMSE (pixels)")
                axis.legend()
                axis.grid(alpha=.2)
            figure.savefig(root / "two_block_curves.pdf")
            figure.clear()
    shard = root / "data/target_holdout/experience.h5"
    if shard.is_file():
        with h5py.File(shard, "r") as handle:
            length = int(handle["ep_len"][0])
            pixels = handle["pixels"][:length:2]
        imageio.imwrite(root / "environment.png", pixels[0])
        imageio.mimsave(root / "collector_rollout.gif", list(pixels), duration=200, loop=0)
    print(f"Review artifacts: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    export_review(parser.parse_args().output_root)
