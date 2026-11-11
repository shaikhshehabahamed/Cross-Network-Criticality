import networkx as nx
import numpy as np
import torch

from src.gnn import graph_to_edge_index
from src.tuning import tune_gnn


def _toy_setup(n=100, seed=0):
    G = nx.gnp_random_graph(n, 0.05, seed=seed, directed=True)
    nodes = list(G.nodes())
    edge_index = graph_to_edge_index(G, nodes)
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(size=(n, 3)).astype("float32"))
    y = torch.from_numpy((x[:, 0] * 2 - x[:, 1]).numpy().astype("float32"))
    return x, edge_index, y


def test_tune_gnn_returns_all_search_space_keys():
    x, edge_index, y = _toy_setup()
    params = tune_gnn(x, edge_index, y, n_trials=3, seed=0)
    assert set(params) == {"hidden_dim", "num_layers", "lr", "weight_decay", "dropout"}
    assert params["hidden_dim"] in [16, 32, 64, 128]
    assert 1 <= params["num_layers"] <= 3
    assert 0.0 <= params["dropout"] <= 0.6


def test_tune_gnn_respects_fixed_params():
    x, edge_index, y = _toy_setup()
    params = tune_gnn(x, edge_index, y, n_trials=3, seed=0, fixed_params={"num_layers": 1})
    assert params["num_layers"] == 1


def test_tune_gnn_is_deterministic_given_seed():
    x, edge_index, y = _toy_setup()
    p1 = tune_gnn(x, edge_index, y, n_trials=4, seed=7)
    p2 = tune_gnn(x, edge_index, y, n_trials=4, seed=7)
    assert p1 == p2
