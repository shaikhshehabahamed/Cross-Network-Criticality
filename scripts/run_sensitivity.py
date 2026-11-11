"""Three robustness/sensitivity checks flagged as open assumptions in
RESEARCH_DIRECTION.md S1.1/S3, each re-running a scoped subset of conditions rather
than the full 8-condition grid (the point is to check whether conclusions hold, not to
re-derive the headline numbers a second time):

1. Edge-direction reversal -- AnonNet's source/target convention isn't documented
   anywhere upstream; both directions were cached from the start (scripts/
   build_ground_truth.py) specifically so this check needs no new simulation.
2. AND vs OR cascade ground truth -- does the choice of propagation rule change which
   method looks best? Also already cached.
3. AnonNet country missingness: impute-as-unknown (the default used everywhere else in
   the pipeline) vs listwise exclusion of the 20.6% of nodes with no resolved country.
   This one DOES need fresh ground truth + features, computed on the induced subgraph
   after exclusion (see src/data.py::subgraph_excluding_missing_country) and cached
   under a distinct network name so it can't collide with or silently overwrite the
   real AnonNet cache.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cascade import run_cascade
from src.data import build_networks, subgraph_excluding_missing_country
from src.experiment import run_condition
from src.predictors import build_structural_features

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "cache" / "ground_truth"
FEAT_DIR = ROOT / "cache" / "features"
OUT_DIR = ROOT / "cache" / "sensitivity"
# Lighter than the main sweep (see run_main_experiment.py) -- these are robustness
# checks on an already-established headline result, not a re-derivation of it.
GNN_TRIALS = 15
GNN_SEEDS = 10


def _run_and_save(nets, train_name, test_name, target, fc, tag, **kwargs):
    out_path = OUT_DIR / f"{tag}__{train_name}_to_{test_name}__{target}__{fc}.pkl"
    if out_path.exists():
        print(f"{out_path.name} cached, skipping")
        return
    print(f"[{tag}] {train_name}->{test_name} / {target} / {fc}")
    result = run_condition(nets, train_name, test_name, target, fc, gnn_trials=GNN_TRIALS, gnn_seeds=GNN_SEEDS, ridge_bootstrap=20, **kwargs)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)


def direction_reversal_check(nets):
    for train_name, test_name in [("AnonNet", "SupplyChainKG"), ("SupplyChainKG", "AnonNet")]:
        _run_and_save(nets, train_name, test_name, "criticality", "topology_only", tag="direction_reversed", direction="reversed")


def and_vs_or_ground_truth_check(nets):
    for train_name, test_name in [("AnonNet", "SupplyChainKG"), ("SupplyChainKG", "AnonNet")]:
        _run_and_save(nets, train_name, test_name, "criticality", "topology_only", tag="cascade_OR", cascade_model="OR")


def build_excluded_anonnet_cache(nets):
    excl = subgraph_excluding_missing_country(nets["AnonNet"])
    name = "AnonNet_excl_missing"
    excl.name = name
    print(f"excluded {excl.meta['excluded_missing_country_nodes']} nodes; {excl.G.number_of_nodes()} remain")

    for direction in ("as_given", "reversed"):
        gt_path = GT_DIR / f"{name}_{direction}.parquet"
        if not gt_path.exists():
            and_res = run_cascade(excl.G, "AND", direction)
            or_res = run_cascade(excl.G, "OR", direction)
            merged = and_res.scores.join(or_res.scores, lsuffix="_and", rsuffix="_or")
            merged.to_parquet(gt_path)

    feat_path = FEAT_DIR / f"{name}.parquet"
    if not feat_path.exists():
        feats = build_structural_features(excl.G, seed=0)
        feats.to_parquet(feat_path)

    return {**nets, name: excl}


def missingness_check(nets_with_excl):
    for test_name in ("SupplyChainKG",):
        _run_and_save(nets_with_excl, "AnonNet_excl_missing", test_name, "criticality", "topology_country", tag="missingness_excluded")
    for train_name in ("SupplyChainKG",):
        _run_and_save(nets_with_excl, train_name, "AnonNet_excl_missing", "criticality", "topology_country", tag="missingness_excluded")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nets = build_networks()
    direction_reversal_check(nets)
    and_vs_or_ground_truth_check(nets)
    nets_with_excl = build_excluded_anonnet_cache(nets)
    missingness_check(nets_with_excl)
    print("sensitivity analyses done")
