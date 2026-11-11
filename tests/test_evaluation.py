import numpy as np
from scipy.stats import spearmanr

from src.evaluation import (
    compute_metrics,
    evaluate_multi_instance,
    evaluate_random_baseline,
    evaluate_single_vector,
    make_shared_resamples,
    summarize,
)
from src.stats import paired_compare


def test_perfect_prediction_gets_top_metrics():
    rng = np.random.default_rng(0)
    y = rng.random(500)
    m = compute_metrics(y, y)
    assert m["spearman"] == 1.0
    assert m["top5pct_overlap"] == 1.0


def test_inverted_prediction_gets_worst_spearman():
    rng = np.random.default_rng(0)
    y = rng.random(500)
    m = compute_metrics(y, -y)
    assert m["spearman"] == -1.0


def test_perfect_predictor_beats_random_in_paired_test():
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.random(n)
    resamples = make_shared_resamples(n, seed=1, n_resamples=200)

    perfect = evaluate_single_vector(y_true, y_true, resamples)
    random_draws = evaluate_random_baseline(y_true, resamples, seed=2)

    result = paired_compare(perfect["spearman"], random_draws["spearman"], "perfect", "random")
    assert result["significant_at_0.05"]
    assert result["mean_diff"] > 0.5


def test_multi_instance_shape_and_range():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.random(n)
    pred_matrix = rng.random((20, n))
    resamples = make_shared_resamples(n, seed=1, n_resamples=100)
    draws = evaluate_multi_instance(y_true, pred_matrix, resamples, seed=3)
    assert draws["spearman"].shape == (100,)
    summary = summarize(draws)
    assert -1.0 <= summary["spearman_mean"] <= 1.0
    assert summary["spearman_ci_lo"] <= summary["spearman_mean"] <= summary["spearman_ci_hi"]


def test_resamples_are_reproducible_given_seed():
    r1 = make_shared_resamples(100, seed=42, n_resamples=10)
    r2 = make_shared_resamples(100, seed=42, n_resamples=10)
    assert np.array_equal(r1, r2)


def test_resamples_have_no_within_row_duplicates():
    resamples = make_shared_resamples(150, seed=7, n_resamples=50)
    for row in resamples:
        assert len(set(row.tolist())) == len(row)


def _naive_reference_single_vector(y_true_full, y_pred_full, resamples):
    """Deliberately slow, obviously-correct per-resample loop using scipy directly --
    the pre-vectorization implementation, kept only as a correctness oracle."""
    spearman = []
    top5 = []
    for idx in resamples:
        yt, yp = y_true_full[idx], y_pred_full[idx]
        spearman.append(spearmanr(yt, yp).statistic)
        k = max(1, int(round(len(yt) * 0.05)))
        true_top = set(np.argpartition(-yt, k - 1)[:k].tolist())
        pred_top = set(np.argpartition(-yp, k - 1)[:k].tolist())
        top5.append(len(true_top & pred_top) / k)
    return np.array(spearman), np.array(top5)


def test_vectorized_matches_naive_reference():
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.normal(size=n)
    y_pred = y_true * 0.6 + rng.normal(size=n) * 0.8  # correlated but noisy, with ties unlikely
    resamples = make_shared_resamples(n, seed=5, n_resamples=60)

    vectorized = evaluate_single_vector(y_true, y_pred, resamples)
    ref_spearman, ref_top5 = _naive_reference_single_vector(y_true, y_pred, resamples)

    np.testing.assert_allclose(vectorized["spearman"], ref_spearman, atol=1e-10)
    np.testing.assert_allclose(vectorized["top5pct_overlap"], ref_top5, atol=1e-10)


def test_vectorized_matches_naive_reference_with_ties():
    """Integer-valued (heavily tied) data, matching the real cascade-count targets --
    rankdata's tie-handling must agree with scipy.stats.spearmanr's."""
    rng = np.random.default_rng(1)
    n = 300
    y_true = rng.integers(0, 5, size=n).astype(float)  # many ties, like real criticality scores
    y_pred = rng.integers(0, 8, size=n).astype(float)
    resamples = make_shared_resamples(n, seed=6, n_resamples=60)

    vectorized = evaluate_single_vector(y_true, y_pred, resamples)
    ref_spearman, ref_top5 = _naive_reference_single_vector(y_true, y_pred, resamples)

    np.testing.assert_allclose(vectorized["spearman"], ref_spearman, atol=1e-10)
    np.testing.assert_allclose(vectorized["top5pct_overlap"], ref_top5, atol=1e-10)
