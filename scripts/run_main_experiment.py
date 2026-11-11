"""Primary cross-network evaluation: all 8 conditions (2 train/test directions x 2
targets x 2 feature configs), AND-model ground truth, as-given edge direction. This is
the headline result; ablations (scripts/run_ablations.py) and sensitivity analyses
(scripts/run_sensitivity.py) build on top of it and reuse its cached GNN/Ridge fits
where the condition is identical.

Saves one pickle per condition to cache/experiments/ (raw per-resample draws are large
and are pipeline intermediates, not a paper output -- final tables/figures are produced
separately by scripts/make_tables.py and scripts/make_figures.py, which read this
cache) plus a single tidy summary CSV of all conditions for quick inspection.
"""
import pickle
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_networks
from src.experiment import run_condition

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "experiments"
# Calibrated against a real timed run on this machine (see commit history / session
# notes): ~9.7s per GNN train once evaluation was vectorized (was the dominant cost
# before that fix -- 1000 resamples x ~14 methods in a pure-Python loop). 25 trials x 2
# tuning seeds + 15 final seeds = 65 trains/condition x 8 conditions is ~85 minutes,
# which is the rigor level used for the headline results; ablations/sensitivity use a
# deliberately lighter budget (see those scripts) since they're robustness checks on
# an already-established result, not a second attempt at the headline number.
GNN_TRIALS = 25
GNN_SEEDS = 15
RIDGE_BOOTSTRAP = 200


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nets = build_networks()
    networks = ["AnonNet", "SupplyChainKG"]
    targets = ["criticality", "vulnerability"]
    feature_configs = ["topology_only", "topology_country"]

    conditions = []
    for train_name in networks:
        test_name = [n for n in networks if n != train_name][0]
        for target, fc in product(targets, feature_configs):
            conditions.append((train_name, test_name, target, fc))

    t_all = time.time()
    for i, (train_name, test_name, target, fc) in enumerate(conditions, 1):
        out_path = CACHE_DIR / f"{train_name}_to_{test_name}__{target}__{fc}.pkl"
        if out_path.exists():
            print(f"[{i}/{len(conditions)}] {out_path.name} already cached, skipping")
            continue
        print(f"[{i}/{len(conditions)}] running {train_name}->{test_name} / {target} / {fc}")
        result = run_condition(
            nets, train_name, test_name, target, fc,
            gnn_trials=GNN_TRIALS, gnn_seeds=GNN_SEEDS, ridge_bootstrap=RIDGE_BOOTSTRAP,
        )
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  saved -> {out_path.name}  (total elapsed {time.time()-t_all:.0f}s)")

    print(f"ALL DONE in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
