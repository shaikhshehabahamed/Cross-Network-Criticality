"""Inductive GraphSAGE regressor for cross-network transfer: trained end-to-end on one
network's full topology with a node-level train/val split for early stopping and
hyperparameter selection, then applied unchanged (same weights, no fine-tuning) to a
completely unseen second network's topology.

Deliberately full-batch (no neighbor sampling) -- both networks are small enough
(<=7,245 nodes) that full-batch message passing is fast and, unlike sampling, exactly
reproducible given a seed.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

# Benchmarked directly on AnonNet (7,245 nodes / 43,922 edges, hidden_dim=64): 1 thread
# 109ms/epoch, 2 threads 63ms/epoch, 4 threads 63ml/epoch, 8 threads 81ms/epoch -- the
# model is far too small for this graph size to benefit from more parallelism, and
# thread-pool dispatch overhead dominates past 2 threads. Fixed rather than left at
# PyTorch's default (which grabs all logical cores) purely for training throughput.
torch.set_num_threads(2)


class SAGERegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.dropout = dropout
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x).squeeze(-1)


def graph_to_edge_index(G: nx.DiGraph, node_order: list) -> torch.Tensor:
    idx = {n: i for i, n in enumerate(node_order)}
    edges = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=np.int64).T
    if edges.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.from_numpy(edges)


def make_train_val_split(n_nodes: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_nodes)
    n_val = max(1, int(n_nodes * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


def train_gnn(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    max_epochs: int = 150,
    patience: int = 15,
):
    torch.manual_seed(seed)
    model = SAGERegressor(x.shape[1], hidden_dim, num_layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    patience_left = patience
    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(x, edge_index)
        loss = F.mse_loss(pred[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index)
            val_loss = F.mse_loss(pred[val_mask], y[val_mask]).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


@torch.no_grad()
def predict(model: SAGERegressor, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x, edge_index).numpy()
