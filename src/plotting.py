"""Shared matplotlib styling for every figure in outputs/figures/, so the whole set
reads as one system rather than each script picking its own defaults. Palette is the
validated colorblind-safe categorical/sequential/status set from the dataviz skill
(references/palette.md), adapted for static print/PDF figures (light surface only --
a journal PDF is not theme-switchable) rather than the skill's web/interactive default.

Colors are assigned by ROLE (best_baseline, ridge, gnn, random, network_a, network_b),
never by raw index, so the same entity keeps the same color across every figure in the
paper -- a reader who learns "GNN is orange" in Figure 3 should not have to relearn it
in Figure 6.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF export is silent otherwise: INFO-logs every subsetted glyph

# Categorical slots 1-8, light-surface steps, validated CVD-safe adjacent ordering.
CAT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}

# Fixed role -> hue assignment, reused identically across every figure.
ROLE = {
    "gnn": CAT["blue"],
    "ridge": CAT["orange"],
    "best_structural": CAT["aqua"],
    "or_reachability": CAT["violet"],
    "random": "#a6a49c",  # muted neutral, deliberately outside the categorical set: a null/chance reference, not a competing method
    "network_a": CAT["blue"],   # AnonNet, wherever two networks are shown side by side
    "network_b": CAT["orange"],  # SupplyChainKG
    "and_model": CAT["blue"],
    "or_model": CAT["violet"],
}

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FIG_DPI = 300


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": FIG_DPI,
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.labelcolor": INK_SECONDARY,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,   # embed as real, editable text -- not outlined paths
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, out_dir, name: str):
    """Every figure saved as both a high-DPI PNG (for quick viewing / drafts) and a
    vector PDF (submission-grade, infinite zoom, embedded real text)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=FIG_DPI)
    fig.savefig(out_dir / f"{name}.pdf")


def percent_formatter():
    return mticker.PercentFormatter(xmax=1.0, decimals=0)


PRETTY_METHOD_NAMES = {
    "gnn": "GNN (GraphSAGE)",
    "ridge": "Ridge (structural features)",
    "random": "Random",
    "in_degree": "In-degree",
    "out_degree": "Out-degree",
    "betweenness": "Betweenness",
    "pagerank_fwd": "PageRank (forward)",
    "pagerank_rev": "PageRank (reversed)",
    "katz_fwd": "Katz (forward)",
    "katz_rev": "Katz (reversed)",
    "k_core": "k-core",
    "criticality_or_reachability": "OR-reachability count",
    "vulnerability_or_reachability": "OR-reachability count",
}


def pretty_name(method: str) -> str:
    return PRETTY_METHOD_NAMES.get(method, method)
