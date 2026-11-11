"""Final consistency pass over every produced table and figure. Run last, after
make_tables.py and make_figures.py. Exits non-zero (and prints every problem found,
not just the first) if anything looks wrong -- this is meant to be the last line of
defense before numbers get quoted in a paper draft.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
GT_DIR = ROOT / "cache" / "ground_truth"

problems = []


def check(condition, message):
    if not condition:
        problems.append(message)


def main():
    expected_tables = [f"table{i}_" for i in range(1, 12)]
    found_tables = {p.stem for p in TABLES.glob("*.csv")}
    for prefix in expected_tables:
        check(any(t.startswith(prefix) for t in found_tables), f"missing table with prefix {prefix}")

    expected_fig_prefixes = [f"fig{i}_" for i in range(1, 11)]
    found_figs = {p.stem for p in FIGURES.glob("*.png")}
    for prefix in expected_fig_prefixes:
        matches = [f for f in found_figs if f.startswith(prefix)]
        check(len(matches) > 0, f"missing figure with prefix {prefix}")
        for m in matches:
            check((FIGURES / f"{m}.pdf").exists(), f"{m} has PNG but no matching PDF")

    for csv_path in TABLES.glob("*.csv"):
        df = pd.read_csv(csv_path)
        check(len(df) > 0, f"{csv_path.name} is empty")
        numeric = df.select_dtypes(include=[np.number])
        check(not numeric.isin([np.inf, -np.inf]).any().any(), f"{csv_path.name} contains inf values")
        for col in numeric.columns:
            if "p_value" in col:
                check(numeric[col].between(0, 1).all(), f"{csv_path.name}:{col} has p-values outside [0,1]")
            if col.startswith("spearman") and "mean" in col:
                check(numeric[col].between(-1, 1).all(), f"{csv_path.name}:{col} has |spearman| > 1")
            if "overlap" in col:
                check(numeric[col].between(0, 1).all(), f"{csv_path.name}:{col} overlap fraction outside [0,1]")

    for gt_path in GT_DIR.glob("*.parquet"):
        gt = pd.read_parquet(gt_path)
        check((gt["criticality_and"] <= gt["criticality_or"]).all(), f"{gt_path.name}: AND>OR violated for criticality")
        check((gt["vulnerability_and"] <= gt["vulnerability_or"]).all(), f"{gt_path.name}: AND>OR violated for vulnerability")
        check(gt["criticality_and"].sum() == gt["vulnerability_and"].sum(), f"{gt_path.name}: AND row/col sum mismatch")
        check(gt["criticality_or"].sum() == gt["vulnerability_or"].sum(), f"{gt_path.name}: OR row/col sum mismatch")
        check((gt >= 0).all().all(), f"{gt_path.name}: negative score present")

    headline_path = next(TABLES.glob("table3_*.csv"), None)
    full_path = next(TABLES.glob("table2_*.csv"), None)
    if headline_path and full_path:
        headline = pd.read_csv(headline_path)
        full = pd.read_csv(full_path)
        for _, row in headline.iterrows():
            if "best_structural_baseline" in row["method"]:
                inner = row["method"].split("(")[1].rstrip(")")
                match = full[(full.train == row.train) & (full.test == row.test) & (full.target == row.target) & (full.feature_config == "topology_only") & (full.method == inner)]
            else:
                match = full[(full.train == row.train) & (full.test == row.test) & (full.target == row.target) & (full.feature_config == "topology_only") & (full.method == row.method)]
            if len(match):
                check(np.isclose(match.iloc[0]["spearman_mean"], row["spearman_mean"], atol=1e-9), f"headline/full mismatch for {row.to_dict()}")

    country_path = next(TABLES.glob("table9_*.csv"), None)
    cross_path = next(TABLES.glob("table11_*.csv"), None)
    if country_path and cross_path:
        country = pd.read_csv(country_path).set_index("country")
        cross = pd.read_csv(cross_path)
        for col in ("n_firms", "criticality_sum", "vulnerability_sum"):
            check((cross[col] >= 0).all(), f"table11: negative {col}")
        check((cross["n_firms"] >= 3).all(), "table11: a row has fewer than MIN_FIRMS_PER_GROUP firms")
        # table11 drops small (country, industry) cells that table9's country-level
        # total still includes -- so the cross-tab total per country can fall short of
        # table9's total, but should never exceed it (that would mean double-counting).
        cross_totals = cross.groupby("country")["criticality_sum"].sum()
        for c, total in cross_totals.items():
            check(total <= country.loc[c, "criticality_sum"] + 1e-6, f"table11 total for {c} exceeds table9's country total")

    if problems:
        print(f"QA FAILED: {len(problems)} problem(s) found\n")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    else:
        print(f"QA PASSED: {len(list(TABLES.glob('*.csv')))} tables, {len(list(FIGURES.glob('*.png')))} figures, all consistent.")


if __name__ == "__main__":
    main()
