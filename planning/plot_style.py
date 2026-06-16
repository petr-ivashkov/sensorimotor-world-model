"""Shared Matplotlib styling for project figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

FULL_WIDTH = 5.5

# Good for two side-by-side panels/subfigures with a small gap.
HALF_WIDTH = 2.65

# Useful default aspect ratio for single-row plots.
GOLDEN_RATIO = (5**0.5 - 1) / 2

palette = {
    # Blues
    "Dark Blue": "#2066a8",
    "Med Blue": "#8ec1da",
    "Light Blue": "#cde1ec",

    # Greys
    "Light Grey": "#ededed",
    "Med Grey": "#c7c7c7",
    "Dark Grey": "#7a7a7a",

    # Reds
    "Light Red": "#f6d6c2",
    "Med Red": "#d47264",
    "Dark Red": "#ae282c",

    # Purples
    "Light Purple": "#D9D2E9",
    "Med Purple": "#7E6AAE",
    "Dark Purple": "#3F2E6D",

    # Basics
    "White": "#ffffff",
    "Black": "#000000",
}

_PAPER_STYLE = {
    # ------------------------------------------------------------------
    # Figure size / export
    # ------------------------------------------------------------------
    "figure.dpi": 150,          # notebook display
    "savefig.dpi": 600,         # publication export
    "savefig.bbox": "standard", # keep exact canvas size
    "savefig.pad_inches": 0.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    # ------------------------------------------------------------------
    # Fonts
    # ------------------------------------------------------------------
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "Liberation Serif",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    # Use editable text in SVG. For final PDF export, fonts are embedded below.
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # ------------------------------------------------------------------
    # Font sizes
    # Main paper text is 10 pt. Plot internals should be slightly smaller
    # but still legible at final size.
    # ------------------------------------------------------------------
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # ------------------------------------------------------------------
    # Lines / markers
    # ------------------------------------------------------------------
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.6,

    # ------------------------------------------------------------------
    # Axes / ticks / grid
    # ------------------------------------------------------------------
    "axes.linewidth": 0.6,
    "axes.edgecolor": "0.25",
    "axes.labelcolor": "black",
    "axes.axisbelow": True,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "xtick.bottom": True,
    "ytick.left": True,

    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,

    "grid.color": "0.88",
    "grid.linewidth": 0.5,

    # ------------------------------------------------------------------
    # Legends
    # ------------------------------------------------------------------
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,

    # Keep layout explicit.
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,

    # Default color cycle.
    "axes.prop_cycle": mpl.cycler(
        color=[
            palette["Dark Blue"],
            palette["Dark Red"],
            palette["Med Purple"],
            palette["Dark Grey"],
            palette["Med Blue"],
            palette["Med Red"],
        ]
    ),
}


def apply_matplotlib_style() -> None:
    """Apply shared plotting defaults for paper figures."""
    mpl.rcParams.update(_PAPER_STYLE)


def figure_size(width: float = FULL_WIDTH, height: float | None = None, ratio: float = GOLDEN_RATIO) -> tuple[float, float]:
    """Return a Matplotlib figsize tuple in inches.

    Parameters
    ----------
    width:
        Figure width in inches.
    height:
        Optional explicit height in inches.
    ratio:
        Height-to-width ratio used when height is not given.
    """
    if height is None:
        height = width * ratio
    return width, height


def savefig(fig: mpl.figure.Figure, path: str, **kwargs) -> None:
    """Save a figure with project defaults."""
    fig.savefig(path, **kwargs)


__all__ = [
    "FULL_WIDTH",
    "HALF_WIDTH",
    "GOLDEN_RATIO",
    "palette",
    "apply_matplotlib_style",
    "figure_size",
    "savefig",
]