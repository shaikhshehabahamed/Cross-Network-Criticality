import networkx as nx
import pandas as pd
import pytest

from src.data import Network
from src.features import FEATURE_CONFIGS, build_feature_matrix


def _toy_network(name="toy", seed=0):
    G = nx.gnp_random_graph(40, 0.1, seed=seed, directed=True)
    countries = (["USA", "JPN", None] * G.number_of_nodes())[: G.number_of_nodes()]
    node_attrs = pd.DataFrame({"country_iso3": countries}, index=list(G.nodes()))
    return Network(name=name, G=G, node_attrs=node_attrs)


def _toy_structural_feats(net, column_order):
    rng = __import__("numpy").random.default_rng(0)
    data = {col: rng.random(net.G.number_of_nodes()) for col in column_order}
    return pd.DataFrame(data, index=list(net.G.nodes()))


@pytest.mark.parametrize("config", FEATURE_CONFIGS)
def test_column_order_is_independent_of_input_order(config):
    """Regression test for the exact bug caught in production: two structural-feature
    caches with the same column SET but different column ORDER (e.g. one alphabetized
    by scripts/build_features.py, one left in insertion order by an ad-hoc caching path
    like scripts/run_sensitivity.py's excluded-network variant) must still produce
    identical, alignable feature matrices."""
    net = _toy_network()
    cols_a = ["in_degree", "out_degree", "betweenness", "pagerank_fwd", "k_core"]
    cols_b = ["k_core", "betweenness", "pagerank_fwd", "in_degree", "out_degree"]  # same set, different order

    struct_a = _toy_structural_feats(net, cols_a)
    struct_b = struct_a[cols_b]  # identical data, columns physically reordered

    feats_a = build_feature_matrix(net, struct_a, config)
    feats_b = build_feature_matrix(net, struct_b, config)

    assert list(feats_a.columns) == list(feats_b.columns)
    pd.testing.assert_frame_equal(feats_a, feats_b)


def test_columns_are_sorted():
    net = _toy_network()
    struct = _toy_structural_feats(net, ["out_degree", "betweenness", "in_degree"])
    feats = build_feature_matrix(net, struct, "topology_only")
    assert list(feats.columns) == sorted(feats.columns)


def test_topology_country_adds_exactly_three_columns():
    net = _toy_network()
    struct = _toy_structural_feats(net, ["in_degree", "out_degree"])
    only = build_feature_matrix(net, struct, "topology_only")
    with_country = build_feature_matrix(net, struct, "topology_country")
    assert set(with_country.columns) - set(only.columns) == {"country_is_top5", "country_log_freq_z", "country_missing"}


def test_row_order_matches_graph_node_order_regardless_of_input_row_order():
    net = _toy_network()
    struct = _toy_structural_feats(net, ["in_degree", "out_degree"])
    shuffled = struct.sample(frac=1.0, random_state=1)  # same rows, shuffled order
    feats = build_feature_matrix(net, shuffled, "topology_only")
    assert list(feats.index) == list(net.G.nodes())
