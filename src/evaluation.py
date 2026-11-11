"""Evaluation metrics and the shared resampling scheme used to attach uncertainty to
every predictor on an equal footing.

Design: for a given (test network, target) condition, draw a fixed set of N_RESAMPLES
random 80% subsamples (without replacement) of the test network's nodes ONCE, and
reuse the identical set of subsamples for every predictor being compared. Because the
subsamples are shared, per-resample metric differences between two methods are paired
observations -- this is what makes the Wilcoxon signed-rank test in src/stats.py valid.

Subsampling without replacement (rather than the bootstrap's with-replacement draws) is
used because the top-k overlap metric is awkward to define over a multiset containing
duplicate nodes; Spearman correlation is computed on the same subsamples for
consistency across metrics, even though it would also be well-defined under an
ordinary bootstrap.

For predictors with multiple trained instances (GNN seeds, Ridge bootstrap fits), a
model instance is drawn at random (with replacement) independently for each resample,
so the resulting distribution reflects both training-side and evaluation-sample-side
uncertainty in one paired set of draws.

Everything below is fully vectorized over all N_RESAMPLES resamples at once (stacked
into 2-D arrays and processed with axis-wise numpy/scipy operations) rather than
looped one resample at a time in Python: an early, correctness-first pure-Python-loop
version was ~100x slower and dominated total pipeline runtime (see git history / the
run_condition timing note). tests/test_evaluation.py::test_vectorized_matches_naive_
reference cross-checks the two against each other on a small case.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

N_RESAMPLES = 1000
SUBSAMPLE_FRAC = 0.8
TOP_K_FRACS = (0.01, 0.05, 0.10)


def make_shared_resamples(n_nodes: int, seed: int, n_resamples: int = N_RESAMPLES) -> np.ndarray:
    """Returns an (n_resamples, n_sub) int array -- every row a distinct random
    subsample of node positions, without replacement within a row."""
    rng = np.random.default_rng(seed)
    n_sub = max(10, int(n_nodes * SUBSAMPLE_FRAC))
    return np.array([rng.choice(n_nodes, size=n_sub, replace=False) for _ in range(n_resamples)])


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation for two (n_resamples, n_sub) arrays."""
    ra = rankdata(a, axis=1, method="average")
    rb = rankdata(b, axis=1, method="average")
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return out


def _top_k_overlap_rows(a: np.ndarray, b: np.ndarray, frac: float) -> np.ndarray:
    """Row-wise top-k overlap fraction for two (n_resamples, n_sub) arrays."""
    n_sub = a.shape[1]
    k = max(1, int(round(n_sub * frac)))

    a_top_idx = np.argpartition(-a, k - 1, axis=1)[:, :k]
    a_mask = np.zeros_like(a, dtype=bool)
    np.put_along_axis(a_mask, a_top_idx, True, axis=1)

    b_top_idx = np.argpartition(-b, k - 1, axis=1)[:, :k]
    b_mask = np.zeros_like(b, dtype=bool)
    np.put_along_axis(b_mask, b_top_idx, True, axis=1)

    return (a_mask & b_mask).sum(axis=1) / k


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Single-pair convenience wrapper (used by tests and one-off checks); treats the
    inputs as one resample of length len(y_true)."""
    yt = y_true.reshape(1, -1)
    yp = y_pred.reshape(1, -1)
    out = {"spearman": float(_spearman_rows(yt, yp)[0])}
    for frac in TOP_K_FRACS:
        out[f"top{int(frac * 100)}pct_overlap"] = float(_top_k_overlap_rows(yt, yp, frac)[0])
    return out


def _metrics_from_stacked(y_true_sub: np.ndarray, y_pred_sub: np.ndarray) -> dict:
    out = {"spearman": _spearman_rows(y_true_sub, y_pred_sub)}
    for frac in TOP_K_FRACS:
        out[f"top{int(frac * 100)}pct_overlap"] = _top_k_overlap_rows(y_true_sub, y_pred_sub, frac)
    return out


def evaluate_single_vector(y_true_full: np.ndarray, y_pred_full: np.ndarray, resamples: np.ndarray) -> dict:
    """One fixed prediction vector (a structural baseline), scored over every shared resample."""
    y_true_sub = y_true_full[resamples]
    y_pred_sub = y_pred_full[resamples]
    return _metrics_from_stacked(y_true_sub, y_pred_sub)


def evaluate_multi_instance(y_true_full: np.ndarray, pred_matrix: np.ndarray, resamples: np.ndarray, seed: int) -> dict:
    """pred_matrix: (n_instances, n_nodes) -- e.g. 20 GNN seeds or 200 Ridge bootstrap
    fits. One instance is drawn per resample so training- and evaluation-randomness are
    both reflected in the single paired distribution used for statistical testing."""
    rng = np.random.default_rng(seed)
    n_instances = pred_matrix.shape[0]
    draws = rng.integers(0, n_instances, size=resamples.shape[0])
    y_true_sub = y_true_full[resamples]
    y_pred_selected = pred_matrix[draws]  # (n_resamples, n_nodes): one instance's full prediction vector per resample
    y_pred_sub = np.take_along_axis(y_pred_selected, resamples, axis=1)
    return _metrics_from_stacked(y_true_sub, y_pred_sub)


def evaluate_random_baseline(y_true_full: np.ndarray, resamples: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true_full)
    y_pred_random = rng.random((resamples.shape[0], n))  # a fresh random ranking per resample
    y_true_sub = y_true_full[resamples]
    y_pred_sub = np.take_along_axis(y_pred_random, resamples, axis=1)
    return _metrics_from_stacked(y_true_sub, y_pred_sub)


def summarize(metric_draws: dict) -> dict:
    """Mean + 95% percentile CI per metric, from the per-resample draw arrays."""
    out = {}
    for k, arr in metric_draws.items():
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_ci_lo"] = float(np.percentile(arr, 2.5))
        out[f"{k}_ci_hi"] = float(np.percentile(arr, 97.5))
    return out
