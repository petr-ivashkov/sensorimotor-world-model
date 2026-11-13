"""Shared, minimal plotting defaults for the analysis notebooks."""

import matplotlib as mpl


def apply_matplotlib_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })
