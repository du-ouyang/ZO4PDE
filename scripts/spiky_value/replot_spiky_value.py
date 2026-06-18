"""Replot the spiky-value diagnostic from a saved pointwise CSV.

This script does not rerun training. It only reads ``spiky_value_pointwise.csv``
and redraws value, gradient, and Hessian curves.

Example:
    cd E:\\doc\\code\\zeroth_order\\program
    python scripts/spiky_value/replot_spiky_value.py

Useful tweaks:
    python scripts/spiky_value/replot_spiky_value.py --lw_exact 1.5 --lw_method 1.2
    python scripts/spiky_value/replot_spiky_value.py --legend_panel gradient --legend_loc upper right
    python scripts/spiky_value/replot_spiky_value.py --xlim -1.5 1.5 --ylim_hessian -4 4
    python scripts/spiky_value/replot_spiky_value.py --layout vertical --fig_height 6.4
    python scripts/spiky_value/replot_spiky_value.py --plot_errors
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PANELS = [
    ("value", "Value"),
    ("gradient", "Gradient"),
    ("hessian", "Hessian"),
]

PROGRAM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROGRAM_ROOT / "scripts/spiky_value/results_spiky_zod_bs16384_eps001/seed42_eps0p01"


def resolve_program_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "scripts":
        return PROGRAM_ROOT / path
    return path


def read_pointwise_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        cols: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in cols:
                cols[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=float) for name, values in cols.items()}


def parse_ylim(raw: list[float] | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    if len(raw) != 2:
        raise ValueError("Axis limit arguments must contain exactly two numbers.")
    return float(raw[0]), float(raw[1])


def set_style(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.titlesize": args.title_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_series(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        available = ", ".join(sorted(data))
        raise KeyError(f"Missing column '{name}'. Available columns: {available}")
    return data[name]


def plot_panel(
    ax,
    data: dict[str, np.ndarray],
    x: np.ndarray,
    quantity: str,
    title: str,
    args: argparse.Namespace,
) -> None:
    true_col = f"true_{quantity}"
    nn_col = f"nn_{quantity}_error" if args.plot_errors else f"nn_{quantity}"
    zod_col = f"{args.zod_prefix}_{quantity}_error" if args.plot_errors else f"{args.zod_prefix}_{quantity}"
    zero = np.zeros_like(x)

    if args.plot_errors:
        ax.axhline(0.0, color="black", linewidth=0.75, alpha=0.45)
        ax.plot(x, get_series(data, nn_col), color=args.nn_color, linewidth=args.lw_method, linestyle="--", label=args.nn_label)
        ax.plot(x, get_series(data, zod_col), color=args.zod_color, linewidth=args.lw_method, linestyle="-", label=args.zod_label)
    else:
        ax.plot(x, get_series(data, true_col), color=args.exact_color, linewidth=args.lw_exact, label=args.exact_label)
        ax.plot(x, get_series(data, nn_col), color=args.nn_color, linewidth=args.lw_method, linestyle="--", label=args.nn_label)
        ax.plot(x, get_series(data, zod_col), color=args.zod_color, linewidth=args.lw_method, linestyle="-", label=args.zod_label)

    if args.show_titles:
        ax.set_title(f"{title} error" if args.plot_errors else title)
    ax.grid(True, color="#D0D0D0", linewidth=0.45, alpha=0.8)

    if args.plot_errors and args.symmetric_error_ylim:
        vals = np.concatenate([get_series(data, nn_col), get_series(data, zod_col), zero])
        max_abs = np.nanmax(np.abs(vals))
        if np.isfinite(max_abs) and max_abs > 0:
            ax.set_ylim(-1.08 * max_abs, 1.08 * max_abs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Result directory containing spiky_value_pointwise.csv. Overrides --csv and, unless --out is set, writes replot output into this directory.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RUN_DIR / "spiky_value_pointwise.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RUN_DIR / "spiky_value_comparison_replot.pdf",
    )
    parser.add_argument("--zod_prefix", type=str, default="zod_multi_point_ls")
    parser.add_argument("--exact_label", type=str, default="Ground truth")
    parser.add_argument("--nn_label", type=str, default="NN-autodiff")
    parser.add_argument("--zod_label", type=str, default="ZOD-m")
    parser.add_argument("--exact_color", type=str, default="black")
    parser.add_argument("--nn_color", type=str, default="#D55E00")
    parser.add_argument("--zod_color", type=str, default="#0072B2")
    parser.add_argument("--lw_exact", type=float, default=1.25)
    parser.add_argument("--lw_method", type=float, default=1.05)
    parser.add_argument("--layout", choices=["horizontal", "vertical"], default="vertical")
    parser.add_argument("--fig_width", type=float, default=6.8)
    parser.add_argument("--fig_height", type=float, default=4.35)
    parser.add_argument("--font_size", type=float, default=8.0)
    parser.add_argument("--title_size", type=float, default=9.0)
    parser.add_argument("--tick_size", type=float, default=7.0)
    parser.add_argument("--legend_size", type=float, default=7.0)
    parser.add_argument("--legend_panel", choices=["figure", "value", "gradient", "hessian", "none"], default="value")
    parser.add_argument("--legend_loc", type=str, default="upper left")
    parser.add_argument("--legend_anchor", type=float, nargs=2)
    parser.add_argument("--legend_ncol", type=int, default=1)
    parser.add_argument("--legend_frame", action="store_true")
    parser.add_argument("--hspace", type=float, default=0.34)
    parser.add_argument("--wspace", type=float, default=0.28)
    parser.add_argument("--top", type=float, default=0.96)
    parser.add_argument("--bottom", type=float, default=0.11)
    parser.add_argument("--left", type=float, default=0.08)
    parser.add_argument("--right", type=float, default=0.98)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--xlim", type=float, nargs=2)
    parser.add_argument("--ylim_value", type=float, nargs=2)
    parser.add_argument("--ylim_gradient", type=float, nargs=2)
    parser.add_argument("--ylim_hessian", type=float, nargs=2)
    parser.add_argument("--plot_errors", action="store_true")
    parser.add_argument("--symmetric_error_ylim", action="store_true")
    parser.add_argument("--hide_titles", action="store_true")
    parser.add_argument("--png", action="store_true", help="Also write a PNG next to the requested output.")
    args = parser.parse_args()
    args.show_titles = not args.hide_titles
    default_out = parser.get_default("out")
    if args.run_dir is not None:
        args.run_dir = resolve_program_path(args.run_dir)
        args.csv = args.run_dir / "spiky_value_pointwise.csv"
        if args.out == default_out:
            args.out = args.run_dir / "spiky_value_comparison_replot.pdf"
    else:
        args.csv = resolve_program_path(args.csv)
    args.out = resolve_program_path(args.out)

    data = read_pointwise_csv(args.csv)
    x = get_series(data, "x")
    set_style(args)

    if args.layout == "horizontal":
        fig, axes = plt.subplots(1, 3, figsize=(args.fig_width, args.fig_height), sharex=False)
    else:
        if args.fig_height == 2.35:
            args.fig_height = 6.4
        fig, axes = plt.subplots(3, 1, figsize=(args.fig_width, args.fig_height), sharex=True)
    for ax, (quantity, title) in zip(axes, PANELS):
        plot_panel(ax, data, x, quantity, title, args)

    if args.xlim is not None:
        for ax in axes:
            ax.set_xlim(*parse_ylim(args.xlim))
    ylims = {
        "value": parse_ylim(args.ylim_value),
        "gradient": parse_ylim(args.ylim_gradient),
        "hessian": parse_ylim(args.ylim_hessian),
    }
    for ax, (quantity, _) in zip(axes, PANELS):
        if ylims[quantity] is not None:
            ax.set_ylim(*ylims[quantity])

    handles, labels = axes[0].get_legend_handles_labels()
    panel_to_axis = {"value": 0, "gradient": 1, "hessian": 2}
    if args.legend_panel == "figure":
        legend_kwargs = {
            "handles": handles,
            "labels": labels,
            "loc": args.legend_loc,
            "frameon": args.legend_frame,
            "ncol": args.legend_ncol,
        }
        if args.legend_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(args.legend_anchor)
        fig.legend(**legend_kwargs)
    elif args.legend_panel != "none":
        legend_kwargs = {
            "loc": args.legend_loc,
            "frameon": args.legend_frame,
            "ncol": args.legend_ncol,
            "handlelength": 2.0,
            "borderpad": 0.25,
            "labelspacing": 0.25,
        }
        if args.legend_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(args.legend_anchor)
        axes[panel_to_axis[args.legend_panel]].legend(**legend_kwargs)

    if args.layout == "horizontal":
        for ax in axes:
            ax.set_xlabel("x")
        fig.subplots_adjust(
            top=args.top,
            bottom=args.bottom,
            left=args.left,
            right=args.right,
            wspace=args.wspace,
        )
    else:
        axes[-1].set_xlabel("x")
        fig.subplots_adjust(
            top=args.top,
            bottom=args.bottom,
            left=args.left,
            right=args.right,
            hspace=args.hspace,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")
    if args.png:
        print(f"Saved {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
