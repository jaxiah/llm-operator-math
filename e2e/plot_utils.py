"""Shared plotting utilities for operator visualization.

Usage:
    from e2e.plot_utils import should_plot, save_fig, setup_style
    from e2e.plot_utils import plot_function_comparison, plot_distribution
    from e2e.plot_utils import plot_heatmap, plot_error_distribution

    if should_plot():
        setup_style()
        fig, ax = plt.subplots()
        ...
        save_fig(fig, __file__, "my_plot")
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def should_plot() -> bool:
    """Check if --plot flag is present in sys.argv."""
    return "--plot" in sys.argv


def setup_style() -> None:
    """Configure matplotlib for clean, publication-quality figures."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.1,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    })
    # Try to use a font that supports CJK characters
    for font in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue


def save_fig(fig: plt.Figure, impl_file: str, name: str) -> str:
    """Save figure as PNG in the same directory as the impl.py file.

    Args:
        fig: matplotlib Figure to save.
        impl_file: __file__ of the calling impl.py (used to resolve directory).
        name: filename without extension.

    Returns:
        Path to the saved PNG file.
    """
    out_dir = os.path.dirname(os.path.abspath(impl_file))
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] saved {path}")
    return path


# ---------------------------------------------------------------------------
# Reusable plot functions
# ---------------------------------------------------------------------------

def plot_function_comparison(
    x: np.ndarray,
    funcs: dict[str, np.ndarray],
    title: str,
    xlabel: str = "x",
    ylabel: str = "y",
    impl_file: str | None = None,
    name: str = "function_comparison",
) -> plt.Figure:
    """Plot multiple functions on the same axes for comparison.

    Args:
        x: shared x-axis values.
        funcs: {label: y_values} dict.
        title, xlabel, ylabel: plot labels.
        impl_file: if provided, auto-save.
        name: filename for saving.
    """
    setup_style()
    fig, ax = plt.subplots()
    for label, y in funcs.items():
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    if impl_file:
        save_fig(fig, impl_file, name)
    return fig


def plot_distribution(
    before: np.ndarray,
    after: np.ndarray,
    title: str,
    labels: tuple[str, str] = ("Before", "After"),
    impl_file: str | None = None,
    name: str = "before_after",
) -> plt.Figure:
    """Plot before/after value distributions as overlapping histograms."""
    setup_style()
    fig, ax = plt.subplots()
    # Flatten and sample if too large
    b = _sample_flat(before)
    a = _sample_flat(after)
    ax.hist(b, bins=100, alpha=0.6, label=labels[0], density=True)
    ax.hist(a, bins=100, alpha=0.6, label=labels[1], density=True)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    if impl_file:
        save_fig(fig, impl_file, name)
    return fig


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    impl_file: str | None = None,
    name: str = "heatmap",
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot a 2D matrix as a heatmap."""
    setup_style()
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if impl_file:
        save_fig(fig, impl_file, name)
    return fig


def plot_error_distribution(
    actual: np.ndarray,
    expected: np.ndarray,
    title: str,
    impl_file: str | None = None,
    name: str = "error_dist",
) -> plt.Figure:
    """Plot histogram of absolute errors between actual and expected."""
    setup_style()
    diff = np.abs(actual.flatten() - expected.flatten()).astype(np.float64)
    fig, ax = plt.subplots()
    # Use log scale for x if errors span many orders of magnitude
    nonzero = diff[diff > 0]
    if len(nonzero) > 0 and nonzero.max() / (nonzero.min() + 1e-30) > 100:
        ax.hist(np.log10(nonzero + 1e-30), bins=100, alpha=0.7, color="steelblue")
        ax.set_xlabel("log10(|error|)")
    else:
        ax.hist(diff, bins=100, alpha=0.7, color="steelblue")
        ax.set_xlabel("|error|")
    ax.set_title(title)
    ax.set_ylabel("Count")
    # Annotate stats
    ax.axvline(np.log10(np.max(diff) + 1e-30) if len(nonzero) > 0 and nonzero.max() / (nonzero.min() + 1e-30) > 100 else np.max(diff),
               color="red", linestyle="--", alpha=0.7, label=f"max={np.max(diff):.2e}")
    ax.axvline(np.log10(np.mean(diff) + 1e-30) if len(nonzero) > 0 and nonzero.max() / (nonzero.min() + 1e-30) > 100 else np.mean(diff),
               color="orange", linestyle="--", alpha=0.7, label=f"mean={np.mean(diff):.2e}")
    ax.legend()
    if impl_file:
        save_fig(fig, impl_file, name)
    return fig


def plot_bar_comparison(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str = "Max Absolute Error",
    impl_file: str | None = None,
    name: str = "component_error",
) -> plt.Figure:
    """Bar chart comparing values across labeled components."""
    setup_style()
    fig, ax = plt.subplots()
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    if impl_file:
        save_fig(fig, impl_file, name)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_flat(arr: np.ndarray, max_points: int = 200_000) -> np.ndarray:
    """Flatten and randomly subsample if too many elements."""
    flat = arr.flatten()
    if len(flat) > max_points:
        rng = np.random.default_rng(42)
        flat = rng.choice(flat, max_points, replace=False)
    return flat.astype(np.float64)
