import numpy as np

from src.ridge_model import bootstrap_ridge_predictions, ridge_feature_importance, tune_ridge_alpha


def _linear_data(n=300, d=5, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_coef = rng.normal(size=d)
    y = X @ true_coef + rng.normal(scale=noise, size=n)
    return X, y, true_coef


def test_tune_ridge_alpha_returns_positive_float():
    X, y, _ = _linear_data()
    alpha = tune_ridge_alpha(X, y)
    assert alpha > 0
    assert np.isfinite(alpha)


def test_bootstrap_predictions_shape_and_finiteness():
    X, y, _ = _linear_data()
    X_test = np.random.default_rng(1).normal(size=(50, X.shape[1]))
    preds = bootstrap_ridge_predictions(X, y, X_test, alpha=1.0, n_bootstrap=30, seed=0)
    assert preds.shape == (30, 50)
    assert np.isfinite(preds).all()


def test_bootstrap_predictions_vary_across_fits():
    """Different bootstrap resamples must actually produce different fitted models --
    a constant-across-rows result would mean the resampling isn't taking effect."""
    X, y, _ = _linear_data(n=200)
    X_test = np.random.default_rng(1).normal(size=(20, X.shape[1]))
    preds = bootstrap_ridge_predictions(X, y, X_test, alpha=0.5, n_bootstrap=30, seed=0)
    assert preds.std(axis=0).mean() > 1e-6


def test_low_alpha_recovers_near_true_coefficients_on_easy_data():
    X, y, true_coef = _linear_data(n=2000, noise=0.01)
    importance = ridge_feature_importance(X, y, alpha=0.01, feature_names=[f"f{i}" for i in range(len(true_coef))])
    fitted = np.array(list(importance.values()))
    # standardized coefficients should at least agree in sign and rough magnitude order
    assert np.corrcoef(fitted, true_coef)[0, 1] > 0.95
