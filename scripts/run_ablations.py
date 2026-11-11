"""Ablation studies, built on top of the main experiment's cached artifacts where
possible:

1. GNN depth (num_layers in {1, 2, 3}), architecture otherwise still tuned by Optuna so
   the comparison isn't confounded by an unfair search budget. Run on both train/test
   directions for the criticality target, topology_only features (the feature-config
   ablation is already the topology_only vs topology_country comparison baked into the
   main sweep, so it's read from that cache in scripts/make_tables.py rather than
   rerun here).
2. Ridge standardized feature importance, one fit per main-sweep condition on the full
   training network (not bootstrapped -- this is a diagnostic, not a point estimate
   used anywhere in the headline results).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import build_networks
from src.experiment import load_ground_truth, load_structural_features, run_condition
from src.features import apply_scaler, build_feature_matrix, fit_scaler_on_train
from src.ridge_model import ridge_feature_importance, tune_ridge_alpha

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "experiments"
OUT_DIR = Path(__file__).resolve().parent.parent / "cache" / "ablations"
# Lighter than the main sweep's 25/15 (see run_main_experiment.py) -- this is a
# robustness check on an already-established headline result, not a re-derivation of
# it, and 3 depth values x 2 directions x 40 trains/run is already a lot of GNN fits.
GNN_TRIALS = 15


def depth_ablation(nets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    directions = [("AnonNet", "SupplyChainKG"), ("SupplyChainKG", "AnonNet")]
    for train_name, test_name in directions:
        for depth in (1, 2, 3):
            out_path = OUT_DIR / f"depth{depth}__{train_name}_to_{test_name}.pkl"
            if out_path.exists():
                print(f"{out_path.name} cached, skipping")
                continue
            print(f"depth ablation: {train_name}->{test_name}, num_layers={depth}")
            result = run_condition(
                nets, train_name, test_name, "criticality", "topology_only",
                gnn_trials=GNN_TRIALS, gnn_seeds=10, ridge_bootstrap=20,
                gnn_fixed_params={"num_layers": depth},
            )
            with open(out_path, "wb") as f:
                pickle.dump(result, f)


def ridge_feature_importance_all_conditions(nets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for pkl_path in sorted(CACHE_DIR.glob("*.pkl")):
        train_name, rest = pkl_path.stem.split("_to_")
        test_name, target, fc = rest.split("__")
        net = nets[train_name]
        struct = load_structural_features(train_name)
        feats = build_feature_matrix(net, struct, fc)
        scaler = fit_scaler_on_train(feats)
        X = apply_scaler(feats, scaler).values.astype(np.float64)
        gt = load_ground_truth(train_name)
        y = np.log1p(gt.loc[feats.index, f"{target}_and"].values)
        alpha = tune_ridge_alpha(X, y)
        importance = ridge_feature_importance(X, y, alpha, list(feats.columns))
        records.append({"train": train_name, "target": target, "feature_config": fc, "alpha": alpha, "importance": importance})
    with open(OUT_DIR / "ridge_feature_importance.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} feature-importance records")


if __name__ == "__main__":
    nets = build_networks()
    depth_ablation(nets)
    ridge_feature_importance_all_conditions(nets)
