"""Paper figure style (palette / DPI / fonts aligned with the previous study)."""

from __future__ import annotations

import matplotlib.pyplot as plt

PAPER_LIGHT_BG = "#FFFFFF"
PAPER_ACCENT = "#2563EB"
PAPER_ORANGE = "#EA580C"
PAPER_GREEN = "#16A34A"
PAPER_RED = "#DC2626"
PAPER_GREY = "#6B7280"
PAPER_PROFILE_COLORS = (PAPER_ACCENT, PAPER_ORANGE, PAPER_GREEN, PAPER_RED)
PAPER_DPI = 200

FAMILY_ORDER = ("cccv", "two_step", "three_step", "pulsed")
FAMILY_LABELS = {
    "cccv": "CCCV",
    "two_step": "Two-step",
    "three_step": "Three-step",
    "pulsed": "Pulsed",
}

GROUP_COLORS = {
    "CCCV 0.5C": "#64748b",
    "CCCV ½C": "#64748b",
    "CCCV 1C": "#2563eb",
    "CCCV 2C": "#1e40af",
    "Random": "#f59e0b",
    "GP-BO": "#16a34a",
    "GP-BO (min Q)": "#0f766e",
    "GP-BO (max R)": "#0f766e",
}

POLICY_STYLE = {
    "CCCV 0.5C": {"color": "#64748b", "ls": "-", "lw": 2.2},
    "CCCV ½C": {"color": "#64748b", "ls": "-", "lw": 2.2},
    "CCCV 1C": {"color": "#2563eb", "ls": "--", "lw": 1.6},
    "CCCV 2C": {"color": "#1e3a8a", "ls": ":", "lw": 1.6},
    "Random": {"color": "#f59e0b", "ls": "-", "lw": 2.4},
    "GP-BO": {"color": "#16a34a", "ls": "-", "lw": 2.6},
    "GP-BO (min Q)": {"color": "#0f766e", "ls": "-", "lw": 2.4},
    "GP-BO (max R)": {"color": "#0f766e", "ls": "--", "lw": 2.2},
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": PAPER_DPI,
        "savefig.dpi": PAPER_DPI,
        "figure.facecolor": PAPER_LIGHT_BG,
        "axes.facecolor": PAPER_LIGHT_BG,
        "savefig.facecolor": PAPER_LIGHT_BG,
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "lines.linewidth": 2.2,
    })


def savefig(fig, path, *, pad_inches: float = 0.25) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=pad_inches,
        facecolor=PAPER_LIGHT_BG, edgecolor=PAPER_LIGHT_BG, transparent=False,
    )
    plt.close(fig)
