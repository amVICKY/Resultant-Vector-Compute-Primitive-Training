"""
visualization/plots.py
----------------------
All plots for Phase 1 of GRH empirical validation.

The plots are designed to tell a specific narrative:
    "Rank-1 is insufficient → subspace approach is required"

Plot 1: rank1_energy per layer        → most bars are low
Plot 2: alignment score per layer     → consistently below threshold
Plot 3: alignment histograms per layer → distributions are wide, not spiked at 1
Plot 4: singular value decay curves   → slow decay = many dims needed
Plot 5: significant rank k_l per layer → k varies significantly, far from 1
Plot 6: summary 2×3 panel             → all evidence in one figure for the paper

Each function saves to results/figures/ and returns the figure object.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from typing import Dict

# ──────────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "primary":   "#2563EB",   # blue — used for main bars/lines
    "secondary": "#DC2626",   # red  — thresholds, warnings
    "ok":        "#16A34A",   # green — values that pass
    "neutral":   "#6B7280",   # gray — baselines, random
    "accent":    "#D97706",   # amber — SVD vs mean contrast
}

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def set_style():
    """Apply consistent matplotlib style for all GRH plots."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#D1D5DB",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#F3F4F6",
        "grid.linewidth":    0.8,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   9,
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
    })


def _save(fig: Figure, filename: str) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path)
    print(f"  Saved → {path}")
    return path


def _short_name(name: str, maxlen: int = 22) -> str:
    """Shorten long layer names for axis labels."""
    return name[-maxlen:] if len(name) > maxlen else name


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Rank-1 Energy Per Layer
# ──────────────────────────────────────────────────────────────────────────────

def plot_rank1_energy(
    results: Dict[str, Dict],
    threshold: float = 0.50,
    save: bool = True
) -> Figure:
    """
    Horizontal bar chart of rank-1 energy per layer.

    Bars below `threshold` are colored red (rank-1 is insufficient).
    Bars above are green (rank-1 might be acceptable).

    This is Plot 1 of the narrative: most layers are red.
    """
    set_style()
    names  = list(results.keys())
    values = [results[n]["rank1_energy"] for n in names]
    short  = [_short_name(n) for n in names]

    colors = [COLORS["ok"] if v >= threshold else COLORS["secondary"] for v in values]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(threshold, color=COLORS["secondary"], linestyle="--",
               linewidth=1.2, label=f"threshold = {threshold}")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Rank-1 Energy  σ₁² / Σσₖ²")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Plot 1 — Rank-1 Energy Per Layer\n"
        "Bars below the threshold → rank-1 approximation is INSUFFICIENT for this layer"
    )
    ax.legend(loc="lower right")

    n_low = sum(1 for v in values if v < threshold)
    ax.annotate(
        f"{n_low}/{len(names)} layers below threshold",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        color=COLORS["secondary"], fontweight="bold"
    )

    fig.tight_layout()
    if save:
        _save(fig, "plot1_rank1_energy.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Alignment Score Per Layer (SVD vs Mean Resultant)
# ──────────────────────────────────────────────────────────────────────────────

def plot_alignment_scores(
    results: Dict[str, Dict],
    threshold: float = 0.70,
    save: bool = True
) -> Figure:
    """
    Grouped bar chart comparing SVD-resultant and mean-resultant alignment per layer.

    If both are low → neither single-vector resultant works.
    SVD is consistently higher than mean → SVD is the better resultant (confirms theory).
    Both consistently below 0.7 → motivates subspace approach.
    """
    set_style()
    names    = list(results.keys())
    svd_vals = [results[n]["mean_cos_svd_resultant"]  for n in names]
    mean_vals= [results[n]["mean_cos_mean_resultant"] for n in names]
    short    = [_short_name(n) for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 5))
    ax.bar(x - width/2, svd_vals,  width, label="SVD resultant (v₁)",
           color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, mean_vals, width, label="Mean resultant (r̄)",
           color=COLORS["accent"],  alpha=0.85, edgecolor="white")
    ax.axhline(threshold, color=COLORS["secondary"], linestyle="--",
               linewidth=1.2, label=f"threshold = {threshold}")

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean |cos θ|  (row alignment to resultant)")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Plot 2 — Row Alignment to Resultant Per Layer\n"
        "Low values → rows do NOT point in a single dominant direction"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    if save:
        _save(fig, "plot2_alignment_scores.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Per-Layer Alignment Histograms
# ──────────────────────────────────────────────────────────────────────────────

def plot_alignment_histograms(
    results: Dict[str, Dict],
    cols: int = 4,
    save: bool = True
) -> Figure:
    """
    Grid of histograms — one per layer — showing the distribution of
    per-row cosine similarities to the SVD resultant.

    A narrow spike at cos=1 → high alignment, rank-1 is valid.
    A wide flat distribution → multiple directions, rank-1 fails.
    This is the most visually diagnostic plot.
    """
    set_style()
    names = list(results.keys())
    n     = len(names)
    rows  = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, name in zip(axes_flat, names):
        cos_vals = results[name]["cos_values"]
        mean_cos = results[name]["mean_cos_svd_resultant"]
        r1e      = results[name]["rank1_energy"]

        color = COLORS["ok"] if mean_cos >= 0.7 else COLORS["secondary"]

        ax.hist(cos_vals, bins=25, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(mean_cos, color="#1F2937", linestyle="--",
                   linewidth=1.0, label=f"μ={mean_cos:.2f}")
        ax.set_xlim(0, 1)
        ax.set_title(_short_name(name, 20), fontsize=8, fontweight="bold")
        ax.set_xlabel("|cos θ|", fontsize=7)
        ax.set_ylabel("count",   fontsize=7)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        ax.annotate(f"E₁={r1e:.2f}", xy=(0.05, 0.88),
                    xycoords="axes fraction", fontsize=7,
                    color=COLORS["secondary"] if r1e < 0.5 else COLORS["ok"])

    # Hide unused subplots
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Plot 3 — Per-Row Alignment Distributions\n"
        "Wide histogram → layer learned multiple independent directions → rank-1 insufficient",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    if save:
        _save(fig, "plot3_alignment_histograms.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4: Singular Value Decay Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_singular_value_decay(
    results: Dict[str, Dict],
    tau: float = 0.90,
    max_curves: int = 16,
    save: bool = True
) -> Figure:
    """
    Overlaid cumulative energy curves for all (or up to max_curves) layers.

    A curve that reaches τ quickly → low significant rank → good compression.
    A curve that rises slowly → high significant rank → layer uses many directions.
    The x-intercept of the τ line on each curve gives k_l for that layer.
    """
    set_style()
    names = list(results.keys())
    if len(names) > max_curves:
        step  = len(names) // max_curves
        names = names[::step][:max_curves]

    cmap   = plt.cm.viridis(np.linspace(0.1, 0.9, len(names)))
    fig, ax = plt.subplots(figsize=(10, 5))

    for color, name in zip(cmap, names):
        energy = results[name]["cumulative_energy"]
        k_range = np.arange(1, len(energy) + 1)
        ax.plot(k_range, energy, color=color, alpha=0.65, linewidth=1.1,
                label=_short_name(name, 18))

    ax.axhline(tau, color=COLORS["secondary"], linestyle="--",
               linewidth=1.3, label=f"τ = {tau}")
    ax.set_xlabel("Rank k (number of singular vectors kept)")
    ax.set_ylabel("Cumulative energy  Σᵢ₌₁ᵏ σᵢ² / Σσᵢ²")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Plot 4 — Singular Value Cumulative Energy Decay\n"
        "Slow rise to τ → layer needs many directions → subspace, not rank-1"
    )
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)

    fig.tight_layout()
    if save:
        _save(fig, "plot4_singular_value_decay.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5: Significant Rank k_l Per Layer
# ──────────────────────────────────────────────────────────────────────────────

def plot_significant_rank(
    results: Dict[str, Dict],
    tau: float = 0.90,
    save: bool = True
) -> Figure:
    """
    Bar chart of k_l (significant rank at τ) per layer.
    Second axis shows k_l / min(m,n) — relative rank.

    Early layers often have high k_l (complex features).
    Late layers often have lower k_l (more concentrated).
    This is Plot 5 of the narrative: k_l is far from 1 everywhere.
    """
    set_style()
    names  = list(results.keys())
    k_vals = [results[n]["significant_rank"]       for n in names]
    ratios = [results[n]["significant_rank_ratio"]  for n in names]
    short  = [_short_name(n) for n in names]

    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(max(10, len(names) * 0.55), 5))
    ax2 = ax1.twinx()

    ax1.bar(x, k_vals, color=COLORS["primary"], alpha=0.75, label=f"k_l at τ={tau}")
    ax2.plot(x, ratios, "o--", color=COLORS["accent"], linewidth=1.2,
             markersize=4, label="k_l / min(m,n)", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=60, ha="right", fontsize=7.5)
    ax1.set_ylabel(f"Significant rank k_l  (τ={tau})", color=COLORS["primary"])
    ax2.set_ylabel("Relative rank  k_l / min(m,n)",   color=COLORS["accent"])
    ax2.set_ylim(0, 1.1)

    ax1.set_title(
        f"Plot 5 — Significant Rank Per Layer  (τ={tau})\n"
        "k_l >> 1 everywhere → subspace dimension, not a single vector"
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    if save:
        _save(fig, "plot5_significant_rank.png")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Plot 6: Revision Summary — All Evidence in One Figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_revision_summary(
    results: Dict[str, Dict],
    tau: float = 0.90,
    rank1_threshold: float = 0.50,
    align_threshold:  float = 0.70,
    save: bool = True
) -> Figure:
    """
    A single 2×3 summary figure with all six pieces of evidence that justify
    replacing the rank-1 hypothesis with the subspace hypothesis.

    Designed to be readable as a standalone figure in a paper or report.

    Panels:
      [0,0]  Rank-1 energy per layer         (main claim: it's low)
      [0,1]  Alignment scores per layer       (corroborates: rows aren't aligned)
      [0,2]  Significant rank k_l per layer   (shows correct dimensionality)
      [1,0]  Singular value decay (sample)    (shows slow decay = multi-dim)
      [1,1]  Alignment histogram (best layer) (what "good" looks like)
      [1,2]  Alignment histogram (worst layer)(what most layers look like)
    """
    set_style()
    names    = list(results.keys())
    r1e_vals = [results[n]["rank1_energy"]            for n in names]
    aln_vals = [results[n]["mean_cos_svd_resultant"]  for n in names]
    k_vals   = [results[n]["significant_rank"]         for n in names]
    short    = [_short_name(n, 16) for n in names]

    # Pick best and worst aligned layers for histogram panels
    best_name  = names[int(np.argmax(aln_vals))]
    worst_name = names[int(np.argmin(aln_vals))]

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    x = np.arange(len(names))

    # [0,0] Rank-1 energy
    colors0 = [COLORS["ok"] if v >= rank1_threshold else COLORS["secondary"] for v in r1e_vals]
    ax00.bar(x, r1e_vals, color=colors0, alpha=0.8, edgecolor="white")
    ax00.axhline(rank1_threshold, color=COLORS["secondary"], linestyle="--", linewidth=1)
    ax00.set_xticks(x); ax00.set_xticklabels(short, rotation=70, ha="right", fontsize=6)
    ax00.set_ylabel("σ₁² / Σσₖ²"); ax00.set_ylim(0, 1.05)
    n_low = sum(1 for v in r1e_vals if v < rank1_threshold)
    ax00.set_title(f"(A) Rank-1 Energy\n{n_low}/{len(names)} layers below {rank1_threshold}")

    # [0,1] Alignment scores
    ax01.bar(x, aln_vals, color=COLORS["primary"], alpha=0.8, edgecolor="white")
    ax01.axhline(align_threshold, color=COLORS["secondary"], linestyle="--", linewidth=1)
    ax01.set_xticks(x); ax01.set_xticklabels(short, rotation=70, ha="right", fontsize=6)
    ax01.set_ylabel("Mean |cos θ| to v₁"); ax01.set_ylim(0, 1.05)
    n_low2 = sum(1 for v in aln_vals if v < align_threshold)
    ax01.set_title(f"(B) Row Alignment to SVD Resultant\n{n_low2}/{len(names)} below {align_threshold}")

    # [0,2] Significant rank k_l
    ax02.bar(x, k_vals, color=COLORS["accent"], alpha=0.8, edgecolor="white")
    ax02.axhline(1, color=COLORS["secondary"], linestyle="--", linewidth=1, label="rank-1 claim")
    ax02.set_xticks(x); ax02.set_xticklabels(short, rotation=70, ha="right", fontsize=6)
    ax02.set_ylabel(f"k_l at τ={tau}"); ax02.legend(fontsize=7)
    ax02.set_title(f"(C) Significant Rank Per Layer\nAll >> 1 → subspace needed")

    # [1,0] Singular value decay (first 8 layers)
    sample_names = names[:8]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_names)))
    for color, name in zip(cmap, sample_names):
        energy = results[name]["cumulative_energy"]
        ax10.plot(np.arange(1, len(energy)+1), energy, color=color,
                  linewidth=1.0, alpha=0.75, label=_short_name(name, 14))
    ax10.axhline(tau, color=COLORS["secondary"], linestyle="--", linewidth=1, label=f"τ={tau}")
    ax10.set_xlabel("Rank k"); ax10.set_ylabel("Cumulative energy")
    ax10.set_ylim(0, 1.05); ax10.legend(fontsize=5.5, ncol=2)
    ax10.set_title("(D) Singular Value Decay (first 8 layers)\nSlow rise → multi-dim structure")

    # [1,1] Histogram best layer
    best_cos = results[best_name]["cos_values"]
    ax11.hist(best_cos, bins=20, color=COLORS["ok"], alpha=0.75, edgecolor="white")
    ax11.axvline(results[best_name]["mean_cos_svd_resultant"],
                 color="#1F2937", linestyle="--", linewidth=1)
    ax11.set_xlim(0, 1); ax11.set_xlabel("|cos θ|"); ax11.set_ylabel("count")
    ax11.set_title(
        f"(E) Best Aligned Layer\n{_short_name(best_name, 22)}\n"
        f"μ={results[best_name]['mean_cos_svd_resultant']:.3f} "
        f"E₁={results[best_name]['rank1_energy']:.3f}"
    )

    # [1,2] Histogram worst layer
    worst_cos = results[worst_name]["cos_values"]
    ax12.hist(worst_cos, bins=20, color=COLORS["secondary"], alpha=0.75, edgecolor="white")
    ax12.axvline(results[worst_name]["mean_cos_svd_resultant"],
                 color="#1F2937", linestyle="--", linewidth=1)
    ax12.set_xlim(0, 1); ax12.set_xlabel("|cos θ|"); ax12.set_ylabel("count")
    ax12.set_title(
        f"(F) Worst Aligned Layer\n{_short_name(worst_name, 22)}\n"
        f"μ={results[worst_name]['mean_cos_svd_resultant']:.3f} "
        f"E₁={results[worst_name]['rank1_energy']:.3f}"
    )

    fig.suptitle(
        "Global Resultant Hypothesis — Phase 1 Evidence\n"
        "Why the rank-1 hypothesis was revised to a subspace hypothesis",
        fontsize=13, fontweight="bold", y=1.01
    )

    if save:
        _save(fig, "plot6_revision_summary.png")
    return fig