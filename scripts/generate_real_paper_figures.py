"""Generate vector figures for the AuralGuard paper using verified HF empirical results.
Outputs:
  - paper/figures/cross_domain_comparison.pdf
  - paper/figures/calibration_vs_discrimination.pdf
  - paper/figures/ci_forest_plot.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Shared IEEE style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "text.usetex": False,
})

# Verified data from master consistency matrix & HF results
MODELS = [
    "B1 (LFCC-LCNN)",
    "B2 (RawNet2)",
    "B3 (AASIST)",
    "B5 (WavLM+OCS)",
    "AuralGuard (v1)",
    "AuralGuard v2",
]

COLORS = {
    "B1 (LFCC-LCNN)":  "#CC79A7",  # purple
    "B2 (RawNet2)":    "#56B4E9",  # sky blue
    "B3 (AASIST)":     "#E69F00",  # orange
    "B5 (WavLM+OCS)":  "#009E73",  # green
    "AuralGuard (v1)": "#0072B2",  # blue
    "AuralGuard v2":   "#D55E00",  # vermilion
}

# Empirical metrics: (EER%, CI_low%, CI_high%, AUROC, ECE, Brier)
METRICS = {
    "B1 (LFCC-LCNN)": {
        "19LA":     (16.29, 15.97, 16.63, 0.9104, 0.2651, 0.1905),
        "ITW":      (50.62, 50.00, 51.19, 0.4979, 0.4292, 0.4354),
        "WaveFake": (45.32, 44.86, 45.76, 0.5638, 0.1965, 0.2079),
    },
    "B2 (RawNet2)": {
        "19LA":     (8.99,  8.72,  9.23,  0.9655, 0.1181, 0.1114),
        "ITW":      (38.08, 37.57, 38.59, 0.6671, 0.5761, 0.5543),
        "WaveFake": (50.37, 49.89, 50.80, 0.4863, 0.1996, 0.2131),
    },
    "B3 (AASIST)": {
        "19LA":     (10.63, 10.36, 10.92, 0.9574, 0.0545, 0.0635),
        "ITW":      (42.33, 41.80, 42.85, 0.6122, 0.5552, 0.5380),
        "WaveFake": (49.89, 49.37, 50.34, 0.4873, 0.2472, 0.2407),
    },
    "B5 (WavLM+OCS)": {
        "19LA":     (2.86,  2.73,  3.00,  0.9845, 0.4487, 0.2664),
        "ITW":      (16.55, 16.12, 16.98, 0.9103, 0.2320, 0.1785),
        "WaveFake": (47.43, 46.91, 47.89, 0.5370, 0.4531, 0.3747),
    },
    "AuralGuard (v1)": {
        "19LA":     (7.30,  7.12,  7.50,  0.9425, 0.4524, 0.2759),
        "ITW":      (23.47, 23.03, 23.89, 0.8345, 0.0965, 0.1966),
        "WaveFake": (48.72, 48.25, 49.22, 0.5207, 0.5048, 0.4252),
    },
    "AuralGuard v2": {
        "19LA":     (6.81,  6.61,  7.00,  0.9498, 0.4746, 0.2991),
        "ITW":      (19.19, 18.78, 19.61, 0.8820, 0.1716, 0.1873),
        "WaveFake": (45.19, 44.69, 45.71, 0.5659, 0.5001, 0.4184),
    },
}


def make_cross_domain_comparison():
    """Bar chart comparing EER across 3 corpora with 95% bootstrap error bars."""
    datasets = ["19LA (In-Domain)", "In-the-Wild (Zero-Shot)", "WaveFake (Zero-Shot)"]
    ds_keys = ["19LA", "ITW", "WaveFake"]
    n_groups = len(datasets)
    n_models = len(MODELS)

    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    bar_width = 0.12
    x = np.arange(n_groups)

    for i, model in enumerate(MODELS):
        eers = [METRICS[model][k][0] for k in ds_keys]
        ci_lows = [METRICS[model][k][0] - METRICS[model][k][1] for k in ds_keys]
        ci_highs = [METRICS[model][k][2] - METRICS[model][k][0] for k in ds_keys]
        yerr = [ci_lows, ci_highs]

        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            eers,
            bar_width,
            yerr=yerr,
            label=model,
            color=COLORS[model],
            edgecolor="black",
            linewidth=0.4,
            capsize=2,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
            zorder=3,
        )

    # Reference chance level line at 50%
    ax.axhline(50.0, color="0.45", linestyle="--", linewidth=0.7, zorder=2)
    ax.text(2.35, 50.8, "Random Chance (50%)", fontsize=6, color="0.45", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight="bold")
    ax.set_ylabel("Equal Error Rate (%)")
    ax.set_ylim(0, 56)
    ax.grid(True, axis="y", color="0.85", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=6,
        frameon=False,
        fontsize=6.5,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "cross_domain_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_calibration_vs_discrimination():
    """Scatter plot: AUROC vs Calibration Quality (1 - ECE) on In-the-Wild."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    for model in MODELS:
        itw_data = METRICS[model]["ITW"]
        auroc = itw_data[3]
        ece = itw_data[4]
        cal_score = 1.0 - ece

        marker = "D" if "AuralGuard" in model else ("s" if "B5" in model else "o")
        size = 65 if "AuralGuard" in model else 45

        ax.scatter(
            auroc,
            cal_score,
            color=COLORS[model],
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
            label=model,
        )

        # Annotate labels with offset to avoid collisions
        dx, dy = 0.015, -0.02
        if "AuralGuard (v1)" in model:
            dx, dy = -0.16, 0.02
        elif "AuralGuard v2" in model:
            dx, dy = -0.15, -0.04
        elif "B5" in model:
            dx, dy = -0.14, 0.02
        elif "B1" in model:
            dx, dy = 0.02, 0.01
        elif "B2" in model:
            dx, dy = 0.02, 0.01
        elif "B3" in model:
            dx, dy = -0.05, -0.04

        ax.annotate(
            model.split()[0],
            (auroc, cal_score),
            textcoords="offset points",
            xytext=(dx * 100, dy * 100),
            fontsize=6,
            fontweight="bold" if "AuralGuard" in model else "normal",
            color=COLORS[model],
        )

    # Highlight ideal quadrant
    ax.axvline(0.80, color="0.75", linestyle=":", linewidth=0.6, zorder=1)
    ax.axhline(0.80, color="0.75", linestyle=":", linewidth=0.6, zorder=1)
    ax.text(
        0.81, 0.94, "High Discrimination &\nHigh Calibration",
        fontsize=5.5, color="#0072B2", style="italic", zorder=2
    )

    ax.set_xlabel("Discrimination on In-the-Wild (AUROC)")
    ax.set_ylabel("Calibration Quality ($1 - \\mathrm{ECE}$)")
    ax.set_xlim(0.45, 0.96)
    ax.set_ylim(0.35, 0.98)
    ax.grid(True, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=5.5, handletextpad=0.2, borderaxespad=0.2)

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / "calibration_vs_discrimination.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_ci_forest_plot():
    """Forest plot of 95% bootstrap confidence intervals for all models and datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    ds_titles = ["ASVspoof 2019 LA", "In-the-Wild", "WaveFake"]
    ds_keys = ["19LA", "ITW", "WaveFake"]

    y_pos = np.arange(len(MODELS))

    for ax, ds_name, ds_key in zip(axes, ds_titles, ds_keys):
        for idx, model in enumerate(MODELS):
            eer, ci_low, ci_high, _, _, _ = METRICS[model][ds_key]
            emph = "AuralGuard" in model

            # Plot horizontal error bar
            ax.errorbar(
                eer,
                idx,
                xerr=[[eer - ci_low], [ci_high - eer]],
                fmt="o" if not emph else "D",
                color=COLORS[model],
                ecolor=COLORS[model],
                elinewidth=1.6 if emph else 1.0,
                capsize=2.5,
                markersize=4.5 if emph else 3.5,
                zorder=4,
            )

        ax.set_title(ds_name, fontsize=7.5, fontweight="bold")
        ax.set_xlabel("EER (%) with 95% CI")
        ax.grid(True, axis="x", color="0.88", zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(MODELS, fontsize=6.5)
    axes[0].invert_yaxis()  # top model at top

    fig.tight_layout(pad=0.3, w_pad=0.8)
    out = FIG_DIR / "ci_forest_plot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    make_cross_domain_comparison()
    make_calibration_vs_discrimination()
    make_ci_forest_plot()


if __name__ == "__main__":
    main()
