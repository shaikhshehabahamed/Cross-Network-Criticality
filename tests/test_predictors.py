import networkx as nx
import numpy as np
import pandas as pd

from src.predictors import build_structural_features


def test_degree_matches_networkx_directly():
    """in_degree/out_degree must match networkx's own in/out degree exactly -- the
    direct correctness check now that reversed-graph degree columns (which would have
    been exact duplicates of these, see predictors.py docstring) are not computed."""
    G = nx.gnp_random_graph(200, 0.02, seed=0, directed=True)
    feats = build_structural_features(G, seed=0)
    expected_in = pd.Series(dict(G.in_degree())).reindex(feats.index)
    expected_out = pd.Series(dict(G.out_degree())).reindex(feats.index)
    pd.testing.assert_series_equal(feats["in_degree"], expected_in, check_names=False)
    pd.testing.assert_series_equal(feats["out_degree"], expected_out, check_names=False)


def test_betweenness_computed_once_not_per_direction():
    G = nx.gnp_random_graph(200, 0.02, seed=0, directed=True)
    feats = build_structural_features(G, seed=0)
    assert "betweenness" in feats.columns
    assert "betweenness_fwd" not in feats.columns
    assert "betweenness_rev" not in feats.columns


def test_no_duplicate_or_perfectly_correlated_columns():
    """Guards against reintroducing the in_degree_fwd == out_degree_rev collinearity
    caught during initial smoke testing (an ill-conditioned-matrix warning from
    RidgeCV): no two feature columns should be identical or perfectly (anti)correlated."""
    G = nx.gnp_random_graph(300, 0.02, seed=0, directed=True)
    feats = build_structural_features(G, seed=0)
    corr = feats.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0.0)
    assert corr.max() < 0.999, corr.max()


def test_columns_identical_across_graphs_of_different_size():
    """Cross-network transfer requires an identical feature schema regardless of
    which network (different size, different degree distribution) produced it."""
    G1 = nx.gnp_random_graph(150, 0.03, seed=1, directed=True)
    G2 = nx.gnp_random_graph(400, 0.01, seed=2, directed=True)
    f1 = build_structural_features(G1, seed=0)
    f2 = build_structural_features(G2, seed=0)
    assert list(f1.columns) == list(f2.columns)


def test_no_missing_values():
    G = nx.gnp_random_graph(100, 0.03, seed=3, directed=True)
    feats = build_structural_features(G, seed=0)
    assert feats.isna().sum().sum() == 0
