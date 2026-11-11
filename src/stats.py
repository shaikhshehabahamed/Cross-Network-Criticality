"""Paired statistical comparison between two predictors evaluated on the same shared
resamples (see src/evaluation.py). Paired because both draw arrays come from the
identical resample sequence, so a plain Wilcoxon signed-rank test on the differences is
appropriate -- no independence assumption between the two samples is needed, only that
the pairing itself is meaningful, which it is by construction here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_compare(metric_a: np.ndarray, metric_b: np.ndarray, label_a: str, label_b: str) -> dict:
    assert metric_a.shape == metric_b.shape
    diff = metric_a - metric_b
    if np.allclose(diff, 0):
        return {
            "a": label_a, "b": label_b, "mean_diff": 0.0,
            "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0, "significant_at_0.05": False,
        }
    stat, p = wilcoxon(diff)
    return {
        "a": label_a,
        "b": label_b,
        "mean_diff": float(diff.mean()),
        "ci_lo": float(np.percentile(diff, 2.5)),
        "ci_hi": float(np.percentile(diff, 97.5)),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Family-wise error control across the multiple pairwise method comparisons run
    per condition, ordered most- to least-significant."""
    order = np.argsort(p_values)
    m = len(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject
