"""Core cross-network experiment: train on one network, evaluate strictly out-of-sample
on the other, for every predictor (structural baselines, random, Ridge, GNN), with
tuning, multi-seed/bootstrap uncertainty, and paired statistical comparisons.

The held-out test network's ground truth is used ONLY inside evaluate_* calls, never
inside tune_gnn/tune_ridge_alpha -- this is the leakage boundary the whole design
depends on; test_qa.py::test_no_test_leakage_in_tuning checks it structurally.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import Network
from src.evaluation import evaluate_multi_instance, evaluate_random_baseline, evaluate_single_vector, make_shared_resamples, summarize
from src.features import build_feature_matrix, apply_scaler, fit_scaler_on_train
from src.gnn import graph_to_edge_index, make_train_val_split, predict, train_gnn
from src.ridge_model import bootstrap_ridge_predictions, tune_ridge_alpha
from src.stats import holm_bonferroni, paired_compare
from src.tuning import tune_gnn

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "cache" / "ground_truth"
FEAT_DIR = ROOT / "cache" / "features"

GNN_N_SEEDS = 20
RIDGE_N_BOOTSTRAP = 200
RESAMPLE_SEED = 12345


def load_ground_truth(name: str, direction: str = "as_given") -> pd.DataFrame:
    return pd.read_parquet(GT_DIR / f"{name}_{direction}.parquet")


def load_structural_features(name: str) -> pd.DataFrame:
    return pd.read_parquet(FEAT_DIR / f"{name}.parquet")


def run_condition(
    nets: dict[str, Network],
    train_name: str,
    test_name: str,
    target: str,               # "criticality" or "vulnerability"
    feature_config: str,       # "topology_only" or "topology_country"
    cascade_model: str = "AND",
    direction: str = "as_given",
    gnn_trials: int = 30,
    gnn_seeds: int = GNN_N_SEEDS,
    ridge_bootstrap: int = RIDGE_N_BOOTSTRAP,
    tuning_seed: int = 0,
    gnn_fixed_params: dict | None = None,
    verbose: bool = True,
) -> dict:
    t_start = time.time()
    train_net, test_net = nets[train_name], nets[test_name]

    gt_train = load_ground_truth(train_name, direction)
    gt_test = load_ground_truth(test_name, direction)
    y_col = f"{target}_{cascade_model.lower()}"

    struct_train = load_structural_features(train_name)
    struct_test = load_structural_features(test_name)

    feat_train_raw = build_feature_matrix(train_net, struct_train, feature_config)
    feat_test_raw = build_feature_matrix(test_net, struct_test, feature_config)
    assert list(feat_train_raw.columns) == list(feat_test_raw.columns)

    scaler = fit_scaler_on_train(feat_train_raw)
    X_train = apply_scaler(feat_train_raw, scaler).values.astype(np.float32)
    X_test = apply_scaler(feat_test_raw, scaler).values.astype(np.float32)

    y_train_log = np.log1p(gt_train.loc[feat_train_raw.index, y_col].values).astype(np.float32)
    y_test_true = gt_test.loc[feat_test_raw.index, y_col].values.astype(np.float64)

    node_order_train = list(train_net.G.nodes())
    node_order_test = list(test_net.G.nodes())
    assert list(feat_train_raw.index) == node_order_train
    assert list(feat_test_raw.index) == node_order_test

    # ---- GNN: tune on train network only, then multi-seed train + OOS predict ----
    x_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train_log)
    edge_index_train = graph_to_edge_index(train_net.G, node_order_train)
    x_test_t = torch.from_numpy(X_test)
    edge_index_test = graph_to_edge_index(test_net.G, node_order_test)

    best_params = tune_gnn(
        x_train_t, edge_index_train, y_train_t, n_trials=gnn_trials, seed=tuning_seed, fixed_params=gnn_fixed_params
    )
    if verbose:
        print(f"  [{train_name}->{test_name} {target}/{feature_config}] best GNN params: {best_params}")

    gnn_preds = np.empty((gnn_seeds, len(node_order_test)), dtype=np.float64)
    for s in range(gnn_seeds):
        train_mask, val_mask = make_train_val_split(len(node_order_train), val_frac=0.2, seed=s)
        model, _ = train_gnn(
            x_train_t, edge_index_train, y_train_t, train_mask, val_mask,
            best_params["hidden_dim"], best_params["num_layers"], best_params["lr"],
            best_params["weight_decay"], best_params["dropout"], seed=s,
        )
        gnn_preds[s] = predict(model, x_test_t, edge_index_test)

    # ---- Ridge: tune alpha on train network only, bootstrap-fit, OOS predict ----
    alpha = tune_ridge_alpha(X_train, y_train_log)
    ridge_preds = bootstrap_ridge_predictions(X_train, y_train_log, X_test, alpha, ridge_bootstrap, seed=1)

    # ---- Structural baselines: no training, direct evaluation on test network ----
    baseline_preds = {col: struct_test.loc[node_order_test, col].values.astype(np.float64) for col in struct_test.columns}
    other_model = "or" if cascade_model == "AND" else "and"
    baseline_preds[f"{target}_{other_model}_reachability"] = gt_test.loc[node_order_test, f"{target}_{other_model}"].values.astype(np.float64)

    # ---- Evaluate everything on shared resamples of the test network ----
    resamples = make_shared_resamples(len(node_order_test), seed=RESAMPLE_SEED)
    all_draws = {}
    for name, vec in baseline_preds.items():
        all_draws[name] = evaluate_single_vector(y_test_true, vec, resamples)
    all_draws["random"] = evaluate_random_baseline(y_test_true, resamples, seed=2)
    all_draws["ridge"] = evaluate_multi_instance(y_test_true, ridge_preds, resamples, seed=3)
    all_draws["gnn"] = evaluate_multi_instance(y_test_true, gnn_preds, resamples, seed=4)

    summaries = {name: summarize(draws) for name, draws in all_draws.items()}

    # ---- Pairwise statistical tests on the primary metric (spearman) ----
    method_names = list(all_draws.keys())
    ranked = sorted(method_names, key=lambda m: -summaries[m]["spearman_mean"])
    pairwise = []
    p_values = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            a, b = ranked[i], ranked[j]
            res = paired_compare(all_draws[a]["spearman"], all_draws[b]["spearman"], a, b)
            pairwise.append(res)
            p_values.append(res["p_value"])

    reject = holm_bonferroni(p_values)
    for res, r in zip(pairwise, reject):
        res["significant_after_holm_bonferroni"] = r

    elapsed = time.time() - t_start
    if verbose:
        top3 = ranked[:3]
        print(f"  done in {elapsed:.1f}s. Ranking (by mean Spearman): " + ", ".join(f"{m}={summaries[m]['spearman_mean']:.3f}" for m in top3) + " ...")

    return {
        "condition": {
            "train": train_name, "test": test_name, "target": target,
            "feature_config": feature_config, "cascade_model": cascade_model, "direction": direction,
        },
        "best_gnn_params": best_params,
        "best_ridge_alpha": alpha,
        "summaries": summaries,
        "pairwise_tests": pairwise,
        "ranking": ranked,
        "elapsed_seconds": elapsed,
    }
