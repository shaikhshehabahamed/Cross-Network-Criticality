"""Every figure for the paper, built on top of the same cached experiment/ablation/
sensitivity pickles that scripts/make_tables.py reads. One shared style
(src/plotting.py) so the full figure set reads as one system: same palette, same role
-> color mapping, same fonts, same PNG+PDF export at every call site.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
from matplotlib.colors import LinearSegmentedColormap

from src.data import build_networks
from src.plotting import CAT, INK, INK_MUTED, ROLE, SEQUENTIAL_BLUE, apply_style, percent_formatter, pretty_name, save_figure

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "cache" / "experiments"
ABL_DIR = ROOT / "cache" / "ablations"
SENS_DIR = ROOT / "cache" / "sensitivity"
GT_DIR = ROOT / "cache" / "ground_truth"
TABLES_DIR = ROOT / "outputs" / "tables"
OUT_DIR = ROOT / "outputs" / "figures"


def _load(pattern, directory=EXP_DIR):
    out = {}
    for p in sorted(directory.glob(pattern)):
        with open(p, "rb") as f:
            out[p.stem] = pickle.load(f)
    return out


def _cond_key(train, test, target, fc):
    return f"{train}_to_{test}__{target}__{fc}"


# ---------------------------------------------------------------------------
def fig1_degree_distribution(nets):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, role in [("AnonNet", "network_a"), ("SupplyChainKG", "network_b")]:
        deg = np.array(sorted(dict(nets[name].G.degree()).values(), reverse=True))
        ccdf_rank = np.arange(1, len(deg) + 1) / len(deg)
        ax.loglog(deg, ccdf_rank, color=ROLE[role], linewidth=1.8, label=name, solid_capstyle="round")
    ax.set_xlabel("Total degree (in + out), log scale")
    ax.set_ylabel("Fraction of nodes with degree $\\geq x$")
    ax.set_title("Degree distribution (CCDF)", loc="left", pad=10)
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.4)
    save_figure(fig, OUT_DIR, "fig1_degree_distribution")
    plt.close(fig)


def fig2_and_vs_or_divergence():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=False)
    for ax, name, role in zip(axes, ["AnonNet", "SupplyChainKG"], ["network_a", "network_b"]):
        gt = pd.read_parquet(GT_DIR / f"{name}_as_given.parquet")
        ax.scatter(gt["criticality_or"] + 1, gt["criticality_and"] + 1, s=6, alpha=0.35, color=ROLE[role], edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("OR-model criticality + 1 (worst case)")
        ax.set_ylabel("AND-model criticality + 1 (redundancy-aware)")
        ax.set_title(name, loc="left")
        lim = [1, max(gt["criticality_or"].max(), gt["criticality_and"].max()) + 1]
        ax.plot(lim, lim, color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.7, label="y = x (no divergence)")
        ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Redundancy assumption changes the criticality picture substantially", x=0.01, ha="left", fontsize=11, y=1.02)
    save_figure(fig, OUT_DIR, "fig2_and_vs_or_divergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
CURATED_METHOD_ORDER = ["random", "criticality_or_reachability", "vulnerability_or_reachability", "best_structural", "ridge", "gnn"]


def _best_structural_row(df):
    baselines = ["in_degree", "out_degree", "betweenness", "pagerank_fwd", "pagerank_rev", "k_core", "katz_fwd", "katz_rev"]
    sub = df[df["method"].isin(baselines)].sort_values("spearman_mean", ascending=False)
    return sub.iloc[0] if len(sub) else None


def fig3_main_results(df_full, target="criticality", fc="topology_only"):
    directions = [("AnonNet", "SupplyChainKG"), ("SupplyChainKG", "AnonNet")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
    for ax, (train, test) in zip(axes, directions):
        sub = df_full[(df_full.train == train) & (df_full.test == test) & (df_full.target == target) & (df_full.feature_config == fc)]
        best_struct = _best_structural_row(sub)
        rows = []
        if best_struct is not None:
            rows.append(("Best structural\n(" + pretty_name(best_struct["method"]) + ")", best_struct, ROLE["best_structural"]))
        for m, label, role in [
            ("ridge", "Ridge", "ridge"),
            ("gnn", "GNN", "gnn"),
            (f"{target}_or_reachability", "OR-reachability", "or_reachability"),
            ("random", "Random", "random"),
        ]:
            r = sub[sub.method == m]
            if len(r):
                rows.append((label, r.iloc[0], ROLE[role]))
        rows = rows[::-1]
        y = np.arange(len(rows))
        for yi, (label, r, color) in zip(y, rows):
            ax.errorbar(
                r["spearman_mean"], yi,
                xerr=[[r["spearman_mean"] - r["spearman_ci_lo"]], [r["spearman_ci_hi"] - r["spearman_mean"]]],
                fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=3, markersize=7,
            )
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.axvline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_xlabel("Spearman correlation with true " + target)
        ax.set_title(f"Train: {train}  →  Test: {test}", loc="left")
    fig.suptitle(f"Out-of-sample {target} prediction, 95% CI over shared resamples ({fc.replace('_', ' ')} features)", x=0.01, ha="left", fontsize=10.5, y=1.03)
    save_figure(fig, OUT_DIR, f"fig3_main_results_{target}")
    plt.close(fig)


def fig4_topk_overlap(df_full, target="criticality", fc="topology_only"):
    directions = [("AnonNet", "SupplyChainKG"), ("SupplyChainKG", "AnonNet")]
    fracs = ["top1pct_overlap_mean", "top5pct_overlap_mean", "top10pct_overlap_mean"]
    frac_labels = ["Top 1%", "Top 5%", "Top 10%"]
    methods = [("best_structural", "Best structural"), ("ridge", "Ridge"), ("gnn", "GNN"), ("random", "Random")]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (train, test) in zip(axes, directions):
        sub = df_full[(df_full.train == train) & (df_full.test == test) & (df_full.target == target) & (df_full.feature_config == fc)]
        best_struct = _best_structural_row(sub)
        width = 0.19
        x = np.arange(len(frac_labels))
        for i, (mkey, mlabel) in enumerate(methods):
            if mkey == "best_structural":
                row = best_struct
            else:
                r = sub[sub.method == mkey]
                row = r.iloc[0] if len(r) else None
            if row is None:
                continue
            vals = [row[f] for f in fracs]
            role = "best_structural" if mkey == "best_structural" else mkey
            ax.bar(x + (i - 1.5) * width, vals, width=width, color=ROLE[role], label=mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(frac_labels)
        ax.yaxis.set_major_formatter(percent_formatter())
        ax.set_ylabel("Overlap with true top-k set")
        ax.set_title(f"{train} → {test}", loc="left")
    axes[0].legend(loc="upper right", ncols=1, fontsize=8.5)
    fig.suptitle(f"Top-k critical-node identification accuracy ({target})", x=0.01, ha="left", fontsize=11, y=1.03)
    save_figure(fig, OUT_DIR, f"fig4_topk_overlap_{target}")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig5_depth_ablation():
    results = _load("depth*.pkl", ABL_DIR)
    if not results:
        print("no depth ablation cache found, skipping fig5")
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    by_dir = {}
    for stem, result in results.items():
        depth = int(stem.split("__")[0].replace("depth", ""))
        cond = result["condition"]
        key = f"{cond['train']}→{cond['test']}"
        by_dir.setdefault(key, {})[depth] = result["summaries"]["gnn"]
    for key, role in zip(sorted(by_dir), ["network_a", "network_b"]):
        depths = sorted(by_dir[key])
        means = [by_dir[key][d]["spearman_mean"] for d in depths]
        los = [by_dir[key][d]["spearman_ci_lo"] for d in depths]
        his = [by_dir[key][d]["spearman_ci_hi"] for d in depths]
        ax.plot(depths, means, marker="o", color=ROLE[role], label=key)
        ax.fill_between(depths, los, his, color=ROLE[role], alpha=0.15, linewidth=0)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("GNN depth (number of SAGEConv layers)")
    ax.set_ylabel("Spearman correlation (criticality, OOS)")
    ax.set_title("GNN depth ablation", loc="left")
    ax.legend(loc="best", fontsize=8.5)
    save_figure(fig, OUT_DIR, "fig5_depth_ablation")
    plt.close(fig)


def fig6_feature_config_ablation(df_full):
    df = df_full[df_full["method"].isin(["ridge", "gnn"])].copy()
    df["direction"] = df["train"] + "→" + df["test"]
    df = df[df["target"] == "criticality"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    directions = sorted(df["direction"].unique())
    methods = ["ridge", "gnn"]
    configs = ["topology_only", "topology_country"]
    width = 0.18
    x = np.arange(len(directions) * len(methods))
    labels = [f"{d}\n{m}" for d in directions for m in methods]
    for ci, (cfg, shade) in enumerate(zip(configs, [0.55, 1.0])):
        vals = []
        for d in directions:
            for m in methods:
                r = df[(df.direction == d) & (df.method == m) & (df.feature_config == cfg)]
                vals.append(r["spearman_mean"].iloc[0] if len(r) else np.nan)
        color = CAT["blue"] if cfg == "topology_only" else CAT["orange"]
        ax.bar(x + (ci - 0.5) * width, vals, width=width, color=color, label=cfg.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Spearman correlation (criticality, OOS)")
    ax.set_title("Feature-configuration ablation: does country help transfer?", loc="left")
    ax.legend(loc="best")
    save_figure(fig, OUT_DIR, "fig6_feature_config_ablation")
    plt.close(fig)


def fig7_sensitivity(df_full):
    primary = df_full[(df_full.target == "criticality") & (df_full.feature_config == "topology_only") & (df_full.method == "gnn")].copy()
    primary["direction"] = primary["train"] + "→" + primary["test"]
    primary["check"] = "Primary (AND, as-given)"

    rows = [primary[["direction", "check", "spearman_mean", "spearman_ci_lo", "spearman_ci_hi"]]]
    labels = {"direction_reversed": "Edges reversed", "cascade_OR": "OR ground truth"}
    for pattern, label in labels.items():
        res = _load(f"{pattern}*.pkl", SENS_DIR)
        recs = []
        for result in res.values():
            cond = result["condition"]
            s = result["summaries"]["gnn"]
            recs.append({"direction": f"{cond['train']}→{cond['test']}", "check": label, **{k: s[k] for k in ["spearman_mean", "spearman_ci_lo", "spearman_ci_hi"]}})
        if recs:
            rows.append(pd.DataFrame(recs))

    df = pd.concat(rows, ignore_index=True)
    if df["check"].nunique() < 2:
        print("not enough sensitivity variants found, skipping fig7")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    checks = list(dict.fromkeys(df["check"]))
    directions = sorted(df["direction"].unique())
    colors = [CAT["blue"], CAT["orange"], CAT["aqua"], CAT["violet"]]
    width = 0.8 / len(checks)
    x = np.arange(len(directions))
    for i, check in enumerate(checks):
        sub = df[df.check == check].set_index("direction").reindex(directions)
        ax.bar(x + (i - (len(checks) - 1) / 2) * width, sub["spearman_mean"], width=width,
               yerr=[sub["spearman_mean"] - sub["spearman_ci_lo"], sub["spearman_ci_hi"] - sub["spearman_mean"]],
               color=colors[i % len(colors)], label=check, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(directions)
    ax.set_ylabel("GNN Spearman correlation (criticality, OOS)")
    ax.set_title("Sensitivity to edge direction and ground-truth definition", loc="left")
    ax.legend(loc="best", fontsize=8.5)
    save_figure(fig, OUT_DIR, "fig7_sensitivity")
    plt.close(fig)


# ---------------------------------------------------------------------------
# pycountry's `.name` is the formal ISO short name, which reads awkwardly in a
# figure ("Korea, Republic of", "Taiwan, Province of China", "Russian Federation").
# Override with the common name for the handful of countries this dataset's top-10
# lists actually surface; everything else falls back to the ISO name unchanged.
_COMMON_NAME_OVERRIDES = {
    "KOR": "South Korea",
    "PRK": "North Korea",
    "TWN": "Taiwan",
    "RUS": "Russia",
    "IRN": "Iran",
    "SYR": "Syria",
    "LAO": "Laos",
    "VNM": "Vietnam",
    "GBR": "United Kingdom",
    "USA": "United States",
    "COD": "DR Congo",
    "COG": "Congo",
}


def _iso3_to_name(code):
    if code in _COMMON_NAME_OVERRIDES:
        return _COMMON_NAME_OVERRIDES[code]
    try:
        return pycountry.countries.get(alpha_3=code).name
    except Exception:
        return code


def _sequential_bar(ax, labels, values, title, xlabel):
    """Shared horizontal-bar rendering for the country/industry exposure charts --
    same sequential-blue-by-rank treatment, same layout, so the two read as a pair."""
    n = len(values)
    colors = [SEQUENTIAL_BLUE[min(int(i / n * len(SEQUENTIAL_BLUE)), len(SEQUENTIAL_BLUE) - 1)] for i in range(n)]
    ax.barh(labels, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


def fig8_country_exposure():
    """Reads table9 (written by scripts/make_tables.py) rather than recomputing the
    aggregation independently -- the figure and the table it's built from must never be
    able to drift apart, so there is exactly one place this number is computed."""
    df = pd.read_csv(TABLES_DIR / "table9_country_exposure.csv").head(10).iloc[::-1]
    names = [_iso3_to_name(c) for c in df["country"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _sequential_bar(
        ax, names, df["criticality_sum"].values,
        "Top 10 countries by aggregate systemic exposure\n(SupplyChainKG)",
        "Aggregate AND-model criticality mass (sum over firms headquartered there)",
    )
    save_figure(fig, OUT_DIR, "fig8_country_exposure")
    plt.close(fig)


def fig9_industry_exposure():
    df = pd.read_csv(TABLES_DIR / "table10_industry_exposure.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _sequential_bar(
        ax, df["industry"], df["criticality_sum"].values,
        "Top 10 industries by aggregate systemic exposure\n(SupplyChainKG)",
        "Aggregate AND-model criticality mass (sum over firms in that industry)",
    )
    save_figure(fig, OUT_DIR, "fig9_industry_exposure")
    plt.close(fig)


def fig10_country_industry_heatmap(n_top=8):
    """The 'which country/industry COMBINATIONS carry disproportionate exposure'
    question from RESEARCH_DIRECTION.md S4.1 -- a single country or industry total can
    hide that its exposure is really concentrated in one specific pairing (e.g. Japan's
    exposure might be almost entirely Automotive, not spread evenly)."""
    cross = pd.read_csv(TABLES_DIR / "table11_country_industry_cross_tab.csv")
    top_countries = pd.read_csv(TABLES_DIR / "table9_country_exposure.csv").head(n_top)["country"].tolist()
    top_industries = pd.read_csv(TABLES_DIR / "table10_industry_exposure.csv").head(n_top)["industry"].tolist()

    sub = cross[cross["country"].isin(top_countries) & cross["industry"].isin(top_industries)]
    matrix = sub.pivot_table(index="industry", columns="country", values="criticality_sum", fill_value=0.0)
    matrix = matrix.reindex(index=top_industries, columns=top_countries, fill_value=0.0)
    country_names = [_iso3_to_name(c) for c in matrix.columns]

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(country_names)))
    ax.set_xticklabels(country_names, rotation=40, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.grid(False)
    ax.spines[:].set_visible(False)

    vmax = matrix.values.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if v <= 0:
                continue
            label_color = "white" if v > vmax * 0.6 else INK
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=label_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Aggregate AND-model criticality mass", color=INK_MUTED)
    cbar.outline.set_visible(False)
    ax.set_title(f"Systemic exposure by country x industry, top {n_top} each\n(SupplyChainKG)", loc="left", pad=12)
    save_figure(fig, OUT_DIR, "fig10_country_industry_heatmap")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nets = build_networks()

    rows = []
    for p in sorted(EXP_DIR.glob("*.pkl")):
        with open(p, "rb") as f:
            result = pickle.load(f)
        cond = result["condition"]
        for method, summary in result["summaries"].items():
            rows.append({**cond, "method": method, **summary})
    df_full = pd.DataFrame(rows)
    if df_full.empty:
        print("No cached experiment results found -- run scripts/run_main_experiment.py first.")
        sys.exit(1)

    fig1_degree_distribution(nets)
    fig2_and_vs_or_divergence()
    fig3_main_results(df_full, "criticality")
    fig3_main_results(df_full, "vulnerability")
    fig4_topk_overlap(df_full, "criticality")
    fig5_depth_ablation()
    fig6_feature_config_ablation(df_full)
    fig7_sensitivity(df_full)
    fig8_country_exposure()
    fig9_industry_exposure()
    fig10_country_industry_heatmap()
    print("ALL FIGURES DONE")
