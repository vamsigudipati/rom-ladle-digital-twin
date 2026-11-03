"""Matplotlib helpers of the notebooks. Nothing here is needed by the models."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Wong colorblind-safe palette (notebook 01) and the Stage 2 palette.
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
CORAL, STEEL, GREEN, GRAY, INK = "#D24726", "#2E5E8C", "#5E9A3F", "#9AA3B0", "#1A1F2E"


def heat(ax, F, title="", cmap="inferno", vmin=None, vmax=None, xlabel="x", ylabel="y"):
    """Heat-map of a scalar field on the unit square (notebook 01 cell 7)."""
    im = ax.imshow(F, extent=[0, 1, 0, 1], origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def heat_ladle(ax, F, title="", cmap="inferno", vmin=None, vmax=None):
    """Heat-map without ticks, as notebook 02 draws the ladle fields (cell 5)."""
    im = ax.imshow(F, origin="lower", extent=[0,1,0,1], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


def panels3(ref, pred, main="", cmap="inferno", show=True):
    """Three panels: reference | prediction | |error| (notebook 01 cell 8). Returns the figure."""
    err = np.abs(pred - ref)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    vmin = float(min(ref.min(), pred.min()))
    vmax = float(max(ref.max(), pred.max()))

    heat(ax[0], ref,  "reference (FOM)", cmap, vmin, vmax)
    heat(ax[1], pred, "ROM prediction",  cmap, vmin, vmax)
    heat(ax[2], err,  f"|error|  (max {err.max():.3g})", "viridis")

    fig.suptitle(main); plt.tight_layout()
    if show:
        plt.show()
    return fig
