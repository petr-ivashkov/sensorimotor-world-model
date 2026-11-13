"""The paper's palette and plotting style, shared by the analysis notebooks."""

import matplotlib as mpl
import numpy as np


FULL_WIDTH = 7.0
BLUE = "#2066a8"
RED = "#ae282c"

# Each method has a light fill and a darker outline, as in the paper.
METHOD_COLORS = {
    "smwm": ("#f6d6c2", RED),
    "sigreg": ("#cde1ec", BLUE),
    "pldm": ("#d1e5d8", "#31754b"),
    "dino_wm": ("#f4e3b1", "#946a0c"),
    "random": ("#ededed", "#7a7a7a"),
    "forward_only": ("#D9D2E9", "#3F2E6D"),
}


def apply_matplotlib_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "0.25",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.2,
        "grid.color": "0.88",
        "grid.linewidth": 0.45,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 170,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def position_colors(xy):
    """Give corresponding state and embedding points the same 2D position color.

    The four corners are blue, green, red, and yellow. Unlike a scalar
    colorbar, this shows how both physical coordinates vary across a plot.
    """
    xy = np.asarray(xy, dtype=float)
    span = np.maximum(np.ptp(xy, axis=0), 1e-12)
    x, y = ((xy - xy.min(axis=0)) / span).T
    bottom_left = np.array([0.10, 0.25, 0.90])
    bottom_right = np.array([0.05, 0.62, 0.25])
    top_left = np.array([0.85, 0.10, 0.10])
    top_right = np.array([0.95, 0.78, 0.10])
    return (
        ((1 - x) * (1 - y))[:, None] * bottom_left
        + (x * (1 - y))[:, None] * bottom_right
        + ((1 - x) * y)[:, None] * top_left
        + (x * y)[:, None] * top_right
    ).clip(0, 1)
