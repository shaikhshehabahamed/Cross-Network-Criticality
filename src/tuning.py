"""Optuna hyperparameter search for the GNN. Only ever sees the training network's own
internal validation split -- the held-out test network is never touched during tuning,
which is asserted by construction (this module takes only train-network tensors).

The objective averages validation MSE over a handful of distinct train/val splits and
weight-init seeds so a single lucky/unlucky split doesn't drive the search.
"""
from __future__ import annotations

import numpy as np
import optuna
import torch

from src.gnn import make_train_val_split, train_gnn

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TUNING_SEEDS = 2


_SEARCH_SPACE = {
    "hidden_dim": lambda t: t.suggest_categorical("hidden_dim", [16, 32, 64, 128]),
    "num_layers": lambda t: t.suggest_int("num_layers", 1, 3),
    "lr": lambda t: t.suggest_float("lr", 1e-4, 5e-2, log=True),
    "weight_decay": lambda t: t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    "dropout": lambda t: t.suggest_float("dropout", 0.0, 0.6),
}


def _objective(trial, x, edge_index, y, n_nodes, fixed_params: dict):
    params = dict(fixed_params)
    for name, suggest in _SEARCH_SPACE.items():
        if name not in fixed_params:
            params[name] = suggest(trial)

    val_losses = []
    for s in range(N_TUNING_SEEDS):
        train_mask, val_mask = make_train_val_split(n_nodes, val_frac=0.2, seed=s)
        _model, best_val = train_gnn(
            x, edge_index, y, train_mask, val_mask,
            params["hidden_dim"], params["num_layers"], params["lr"],
            params["weight_decay"], params["dropout"], seed=s,
        )
        val_losses.append(best_val)
    return float(np.mean(val_losses))


def tune_gnn(
    x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor, n_trials: int, seed: int, fixed_params: dict | None = None
) -> dict:
    """fixed_params: e.g. {"num_layers": 1} to hold an architecture choice fixed while
    the remaining hyperparameters are still tuned -- used by the GNN-depth ablation so
    the comparison isn't confounded by depth-1 and depth-3 also getting different lr/
    dropout budgets from an unfair search."""
    fixed_params = fixed_params or {}
    n_nodes = x.shape[0]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda t: _objective(t, x, edge_index, y, n_nodes, fixed_params), n_trials=n_trials, show_progress_bar=False
    )
    return {**fixed_params, **study.best_params}
