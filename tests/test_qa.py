"""Structural checks on the experiment pipeline itself (not the data) -- these guard
the single most important methodological property of the whole design: the held-out
test network's ground truth must never influence tuning or training.
"""
import inspect

from src import experiment, tuning


def test_tune_gnn_signature_has_no_test_network_argument():
    sig = inspect.signature(tuning.tune_gnn)
    # fixed_params only ever carries architecture choices (e.g. {"num_layers": 1} for
    # the depth ablation) -- never data -- so it doesn't reopen the leakage boundary.
    assert set(sig.parameters) == {"x", "edge_index", "y", "n_trials", "seed", "fixed_params"}


def test_tune_ridge_alpha_signature_has_no_test_network_argument():
    from src.ridge_model import tune_ridge_alpha

    sig = inspect.signature(tune_ridge_alpha)
    assert set(sig.parameters) == {"X_train", "y_train", "cv"}


def test_run_condition_builds_gnn_tuning_inputs_from_train_network_only():
    """Static check on the source: the tune_gnn(...) call inside run_condition must be
    constructed only from *_train / *train* identifiers, never from test-network
    variables. A crude but effective guard against a future edit introducing leakage."""
    src = inspect.getsource(experiment.run_condition)
    call_line = next(line for line in src.splitlines() if "tune_gnn(" in line and "def " not in line)
    assert "test" not in call_line.lower(), call_line
