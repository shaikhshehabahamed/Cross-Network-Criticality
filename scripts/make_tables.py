"""Assemble every paper-ready CSV table from the cached experiment/ablation/
sensitivity/ridge-importance pickles. Run after run_main_experiment.py,
run_ablations.py, and run_sensitivity.py have all completed.

Only final, formatted tables are written to outputs/tables/ -- the per-resample raw
draws stay in cache/, which is a pipeline intermediate, not a paper deliverable.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import pandas as pd

from src.data import build_networks

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "cache" / "experiments"
ABL_DIR = ROOT / "cache" / "ablations"
SENS_DIR = ROOT / "cache" / "sensitivity"
GT_DIR = ROOT / "cache" / "ground_truth"
OUT_DIR = ROOT / "outputs" / "tables"

MIN_FIRMS_PER_GROUP = 3  # groups smaller than this are dropped from exposure tables: a
# mean over 1-2 firms is not a meaningful "typical firm in this group" estimate and
# mostly just adds noisy rows


def load_all(pattern, directory=EXP_DIR):
    results = []
    for p in sorted(directory.glob(pattern)):
        with open(p, "rb") as f:
            results.append((p.stem, pickle.load(f)))
    return results


def table_network_stats(nets):
    rows = []
    for net in nets.values():
        Gu = net.G.to_undirected()
        sccs = sorted((len(c) for c in nx.strongly_connected_components(net.G)), reverse=True)
        rows.append({
            "network": net.name,
            "n_nodes": net.G.number_of_nodes(),
            "n_edges": net.G.number_of_edges(),
            "density": nx.density(net.G),
            "mean_degree": sum(dict(net.G.degree()).values()) / net.G.number_of_nodes(),
            "avg_clustering_undirected": nx.average_clustering(Gu),
            "degree_assortativity": nx.degree_assortativity_coefficient(net.G),
            "n_weakly_connected_components": nx.number_weakly_connected_components(net.G),
            "largest_scc_frac": sccs[0] / net.G.number_of_nodes(),
            "country_missing_frac": float(net.node_attrs["country_iso3"].isna().mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table1_network_descriptive_stats.csv", index=False)
    print("wrote table1_network_descriptive_stats.csv")


def table_main_results():
    rows = []
    for stem, result in load_all("*.pkl"):
        cond = result["condition"]
        for method, summary in result["summaries"].items():
            row = {**cond, "method": method}
            row.update(summary)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table2_full_results_all_methods.csv", index=False)
    print(f"wrote table2_full_results_all_methods.csv ({len(df)} rows)")
    return df


def table_headline_results(df_full):
    """Curated subset for the paper body: best single structural baseline, ridge, gnn,
    random -- topology_only feature config, primary (AND, as_given) ground truth."""
    keep_methods_baselines = ["in_degree", "out_degree", "betweenness", "pagerank_fwd", "pagerank_rev", "k_core", "katz_fwd", "katz_rev"]
    df = df_full[df_full["feature_config"] == "topology_only"].copy()
    best_baseline_rows = (
        df[df["method"].isin(keep_methods_baselines)]
        .sort_values("spearman_mean", ascending=False)
        .groupby(["train", "test", "target"], as_index=False)
        .first()
    )
    best_baseline_rows["method"] = "best_structural_baseline (" + best_baseline_rows["method"] + ")"
    headline = pd.concat([
        best_baseline_rows,
        df[df["method"].isin(["ridge", "gnn", "random", "criticality_or_reachability", "vulnerability_or_reachability"])],
    ])
    cols = ["train", "test", "target", "method", "spearman_mean", "spearman_ci_lo", "spearman_ci_hi",
            "top1pct_overlap_mean", "top5pct_overlap_mean", "top10pct_overlap_mean"]
    headline = headline[cols].sort_values(["train", "test", "target", "spearman_mean"], ascending=[True, True, True, False])
    headline.to_csv(OUT_DIR / "table3_headline_results.csv", index=False)
    print(f"wrote table3_headline_results.csv ({len(headline)} rows)")


def table_pairwise_tests():
    rows = []
    for stem, result in load_all("*.pkl"):
        cond = result["condition"]
        for test in result["pairwise_tests"]:
            rows.append({**cond, **test})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table4_pairwise_statistical_tests.csv", index=False)
    print(f"wrote table4_pairwise_statistical_tests.csv ({len(df)} rows)")


def table_depth_ablation():
    rows = []
    for stem, result in load_all("depth*.pkl", ABL_DIR):
        depth = int(stem.split("__")[0].replace("depth", ""))
        cond = result["condition"]
        gnn_summary = result["summaries"]["gnn"]
        rows.append({**cond, "num_layers": depth, **gnn_summary, "best_params": json.dumps(result["best_gnn_params"])})
    if not rows:
        print("no depth ablation results found, skipping")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table5_gnn_depth_ablation.csv", index=False)
    print(f"wrote table5_gnn_depth_ablation.csv ({len(df)} rows)")


def table_feature_config_ablation(df_full):
    df = df_full[df_full["method"].isin(["ridge", "gnn"])].copy()
    pivot = df.pivot_table(index=["train", "test", "target", "method"], columns="feature_config", values="spearman_mean").reset_index()
    if "topology_only" in pivot.columns and "topology_country" in pivot.columns:
        pivot["delta_country_minus_topology_only"] = pivot["topology_country"] - pivot["topology_only"]
    pivot.to_csv(OUT_DIR / "table6_feature_config_ablation.csv", index=False)
    print(f"wrote table6_feature_config_ablation.csv ({len(pivot)} rows)")


def table_ridge_feature_importance():
    path = ABL_DIR / "ridge_feature_importance.json"
    if not path.exists():
        print("no ridge feature importance file found, skipping")
        return
    records = json.loads(path.read_text())
    rows = []
    for rec in records:
        for feat, coef in rec["importance"].items():
            rows.append({"train": rec["train"], "target": rec["target"], "feature_config": rec["feature_config"], "alpha": rec["alpha"], "feature": feat, "standardized_coef": coef})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table7_ridge_feature_importance.csv", index=False)
    print(f"wrote table7_ridge_feature_importance.csv ({len(df)} rows)")


def table_sensitivity():
    rows = []
    for pattern in ["direction_reversed*.pkl", "cascade_OR*.pkl", "missingness_excluded*.pkl"]:
        for stem, result in load_all(pattern, SENS_DIR):
            tag = stem.split("__")[0]
            cond = result["condition"]
            for method in ("gnn", "ridge"):
                if method in result["summaries"]:
                    rows.append({"sensitivity_check": tag, **cond, "method": method, **result["summaries"][method]})
            best = result["ranking"][0]
            rows.append({"sensitivity_check": tag, **cond, "method": f"top_ranked ({best})", **result["summaries"][best]})
    if not rows:
        print("no sensitivity results found, skipping")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table8_sensitivity_analysis.csv", index=False)
    print(f"wrote table8_sensitivity_analysis.csv ({len(df)} rows)")


def _exposure_by_group(net, gt, group_col: str) -> pd.DataFrame:
    """Aggregate criticality/vulnerability mass by a single node attribute (country or
    industry). `sum` measures total systemic mass concentrated in the group (driven by
    both firm count and per-firm risk); `mean` isolates the per-firm effect so a group
    doesn't look disproportionately important purely because it has many firms in it --
    both are reported since a Q1 reviewer would ask for the one I omitted."""
    df = net.node_attrs[[group_col]].join(gt[["criticality_and", "vulnerability_and"]])
    df = df.dropna(subset=[group_col])
    agg = df.groupby(group_col).agg(
        n_firms=("criticality_and", "size"),
        criticality_sum=("criticality_and", "sum"),
        criticality_mean=("criticality_and", "mean"),
        vulnerability_sum=("vulnerability_and", "sum"),
        vulnerability_mean=("vulnerability_and", "mean"),
    )
    agg = agg[agg["n_firms"] >= MIN_FIRMS_PER_GROUP]
    return agg.sort_values("criticality_sum", ascending=False).reset_index()


def table_country_exposure(net):
    gt = pd.read_parquet(GT_DIR / "SupplyChainKG_as_given.parquet")
    df = _exposure_by_group(net, gt, "country_iso3").rename(columns={"country_iso3": "country"})
    df.to_csv(OUT_DIR / "table9_country_exposure.csv", index=False)
    print(f"wrote table9_country_exposure.csv ({len(df)} rows)")


def table_industry_exposure(net):
    gt = pd.read_parquet(GT_DIR / "SupplyChainKG_as_given.parquet")
    df = _exposure_by_group(net, gt, "industry_raw").rename(columns={"industry_raw": "industry"})
    df.to_csv(OUT_DIR / "table10_industry_exposure.csv", index=False)
    print(f"wrote table10_industry_exposure.csv ({len(df)} rows)")


def table_country_industry_cross_tab(net):
    """Every non-empty (country, industry) combination -- the full data behind the
    'which combinations carry disproportionate exposure' question from RESEARCH_
    DIRECTION.md S4.1; fig10 plots only the top slice of this for readability, but the
    complete cross-tab belongs in the paper's supplementary material, not just the figure."""
    gt = pd.read_parquet(GT_DIR / "SupplyChainKG_as_given.parquet")
    df = net.node_attrs[["country_iso3", "industry_raw"]].join(gt[["criticality_and", "vulnerability_and"]])
    df = df.dropna(subset=["country_iso3"])
    agg = (
        df.groupby(["country_iso3", "industry_raw"])
        .agg(
            n_firms=("criticality_and", "size"),
            criticality_sum=("criticality_and", "sum"),
            vulnerability_sum=("vulnerability_and", "sum"),
        )
        .reset_index()
        .rename(columns={"country_iso3": "country", "industry_raw": "industry"})
    )
    agg = agg[agg["n_firms"] >= MIN_FIRMS_PER_GROUP]
    agg = agg.sort_values("criticality_sum", ascending=False)
    agg.to_csv(OUT_DIR / "table11_country_industry_cross_tab.csv", index=False)
    print(f"wrote table11_country_industry_cross_tab.csv ({len(agg)} rows)")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nets = build_networks()
    table_network_stats(nets)
    df_full = table_main_results()
    table_headline_results(df_full)
    table_pairwise_tests()
    table_depth_ablation()
    table_feature_config_ablation(df_full)
    table_ridge_feature_importance()
    table_sensitivity()
    table_country_exposure(nets["SupplyChainKG"])
    table_industry_exposure(nets["SupplyChainKG"])
    table_country_industry_cross_tab(nets["SupplyChainKG"])
    print("ALL TABLES DONE")
