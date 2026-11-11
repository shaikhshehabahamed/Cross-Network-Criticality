import networkx as nx
import numpy as np
import torch

from src.gnn import graph_to_edge_index, make_train_val_split, predict, train_gnn


def _toy_graph(n=120, seed=0):
    G = nx.gnp_random_graph(n, 0.05, seed=seed, directed=True)
    return G, list(G.nodes())


def test_train_val_split_partitions_without_overlap():
    train_mask, val_mask = make_train_val_split(100, val_frac=0.2, seed=0)
    assert (train_mask & val_mask).sum() == 0
    assert train_mask.sum() + val_mask.sum() == 100
    assert 15 <= val_mask.sum() <= 25


def test_gnn_learns_a_trivial_linear_signal():
    """Target = 3*feature0 - feature1, both features random. If the model cannot beat
    a no-learning baseline on this trivial task, something in the training loop is
    broken (wrong loss, frozen weights, wrong mask indexing, etc.)."""
    G, nodes = _toy_graph(n=150, seed=0)
    n = len(nodes)
    edge_index = graph_to_edge_index(G, nodes)
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=(n, 4)).astype(np.float32)
    y_np = (3 * x_np[:, 0] - x_np[:, 1]).astype(np.float32)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    train_mask, val_mask = make_train_val_split(n, val_frac=0.3, seed=0)

    baseline_val_mse = float(((y[val_mask] - y[train_mask].mean()) ** 2).mean())
    _model, best_val = train_gnn(
        x, edge_index, y, train_mask, val_mask,
        hidden_dim=32, num_layers=2, lr=0.01, weight_decay=0.0, dropout=0.0, seed=0,
        max_epochs=300, patience=30,
    )
    assert best_val < 0.5 * baseline_val_mse


def test_predict_transfers_to_a_different_sized_graph():
    """The whole point of this model: train on one graph, predict on a structurally
    unrelated graph of a different size, with no retraining."""
    G1, nodes1 = _toy_graph(n=100, seed=1)
    G2, nodes2 = _toy_graph(n=250, seed=2)
    edge_index1 = graph_to_edge_index(G1, nodes1)
    edge_index2 = graph_to_edge_index(G2, nodes2)

    rng = np.random.default_rng(0)
    x1 = torch.from_numpy(rng.normal(size=(100, 3)).astype("float32"))
    y1 = torch.from_numpy(rng.normal(size=100).astype("float32"))
    x2 = torch.from_numpy(rng.normal(size=(250, 3)).astype("float32"))

    train_mask, val_mask = make_train_val_split(100, val_frac=0.2, seed=0)
    model, _ = train_gnn(
        x1, edge_index1, y1, train_mask, val_mask,
        hidden_dim=16, num_layers=2, lr=0.01, weight_decay=0.0, dropout=0.0, seed=0,
        max_epochs=20, patience=20,
    )
    preds = predict(model, x2, edge_index2)
    assert preds.shape == (250,)
    assert np.isfinite(preds).all()


def test_seed_determinism():
    G, nodes = _toy_graph(n=80, seed=3)
    edge_index = graph_to_edge_index(G, nodes)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(80, 3)).astype("float32"))
    y = torch.from_numpy(rng.normal(size=80).astype("float32"))
    train_mask, val_mask = make_train_val_split(80, val_frac=0.2, seed=0)

    _m1, v1 = train_gnn(x, edge_index, y, train_mask, val_mask, 16, 2, 0.01, 0.0, 0.0, seed=42, max_epochs=30, patience=30)
    _m2, v2 = train_gnn(x, edge_index, y, train_mask, val_mask, 16, 2, 0.01, 0.0, 0.0, seed=42, max_epochs=30, patience=30)
    assert v1 == v2
