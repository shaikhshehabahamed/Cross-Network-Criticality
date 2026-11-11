"""Compute and cache the structural feature suite for both networks (~2 min, dominated
by approximate betweenness) so downstream tuning/training/ablation runs never recompute
it. Also enforces a common column schema across both networks -- required for
cross-network model transfer -- rather than assuming it (see src/features.py docstring).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_networks
from src.predictors import build_structural_features

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "features"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nets = build_networks()
    feats = {name: build_structural_features(net.G, seed=0) for name, net in nets.items()}

    common_cols = sorted(set.intersection(*(set(f.columns) for f in feats.values())))
    dropped = {name: sorted(set(f.columns) - set(common_cols)) for name, f in feats.items()}
    if any(dropped.values()):
        print(f"WARNING: non-common columns dropped for schema alignment: {dropped}")

    for name, f in feats.items():
        f = f[common_cols]
        out_path = CACHE_DIR / f"{name}.parquet"
        f.to_parquet(out_path)
        print(f"{name:14s} -> {out_path.name}  ({f.shape[0]} nodes x {f.shape[1]} features)")

    assert all(list(feats[n][common_cols].columns) == common_cols for n in feats)
    print("common feature schema:", common_cols)


if __name__ == "__main__":
    main()
