"""Ridge-regression predictor: the interpretable, cheap-to-fit contender against the
GNN. Alpha is selected by cross-validation on the training network only; "multi-seed"
uncertainty is obtained by refitting on bootstrap resamples of the training network's
nodes (the deterministic-model analogue of the GNN's random weight-init seeds).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV


def tune_ridge_alpha(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> float:
    alphas = np.logspace(-3, 3, 25)
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    return float(model.alpha_)


def bootstrap_ridge_predictions(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float, n_bootstrap: int, seed: int
) -> np.ndarray:
    """Returns an (n_bootstrap, n_test_nodes) array of predictions."""
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    preds = np.empty((n_bootstrap, X_test.shape[0]), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        model = Ridge(alpha=alpha)
        model.fit(X_train[idx], y_train[idx])
        preds[b] = model.predict(X_test)
    return preds


def ridge_feature_importance(X_train: np.ndarray, y_train: np.ndarray, alpha: float, feature_names: list) -> dict:
    """Standardized coefficients from a single fit on the full training network --
    used for the feature-importance ablation, not for prediction."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return dict(zip(feature_names, model.coef_.tolist()))
