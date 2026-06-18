"""Miscellaneous utilities."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})


def plot_error_history(csv_path: str, save_dir: Optional[str] = None, relative: bool = True):
    """Plot error curves from a saved ``error_history.csv``."""
    df = pd.read_csv(csv_path)
    iters = df["iteration"].values

    if relative:
        v = df["value_relative_rmse"].values
        g = df["gradient_relative_rmse"].values
        h = df["hessian_relative_rmse"].values
        ylabel = "Relative RMSE"
    else:
        v = df["value_error"].values
        g = df["gradient_error"].values
        h = df["hessian_error"].values
        ylabel = "Mean Path Integral Error"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(iters, v, "o-", color="steelblue", lw=2, ms=6)
    axes[0].set(title="Value Error", xlabel="Iteration", ylabel=ylabel, yscale="log")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, g, "s-", color="seagreen", lw=2, ms=6)
    axes[1].set(title="Gradient Error", xlabel="Iteration", ylabel=ylabel, yscale="log")
    axes[1].grid(alpha=0.3)

    axes[2].plot(iters, h, "^-", color="darkorange", lw=2, ms=6)
    axes[2].set(title="Hessian Error", xlabel="Iteration", ylabel=ylabel, yscale="log")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    tag = "relative" if relative else "absolute"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(save_dir, f"error_history_{tag}.{ext}"), dpi=300, bbox_inches="tight")
    plt.show()
    return fig
