"""Generate and cache cascade ground truth (criticality, vulnerability) for both
networks, both edge-direction conventions, and both propagation models.

Output: cache/ground_truth/{network}_{direction}.parquet, columns:
    criticality_and, vulnerability_and, criticality_or, vulnerability_or
indexed by the network's native node id.

Cheap to recompute (~0.2-0.4s per condition) -- this cache exists to decouple pipeline
stages, not because recomputation is expensive.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cascade import run_cascade
from src.data import build_networks

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "ground_truth"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    nets = build_networks()
    for net in nets.values():
        for direction in ("as_given", "reversed"):
            and_res = run_cascade(net.G, "AND", direction)
            or_res = run_cascade(net.G, "OR", direction)
            merged = and_res.scores.join(or_res.scores, lsuffix="_and", rsuffix="_or")
            merged = merged.reindex(list(net.G.nodes()))  # fixed, deterministic row order
            assert merged.isna().sum().sum() == 0, "ground truth has unexpected gaps"
            assert (merged["criticality_and"] <= merged["criticality_or"]).all()
            assert (merged["vulnerability_and"] <= merged["vulnerability_or"]).all()

            out_path = CACHE_DIR / f"{net.name}_{direction}.parquet"
            merged.to_parquet(out_path)
            print(
                f"{net.name:14s} {direction:9s} -> {out_path.name}  "
                f"(AND crit max={merged['criticality_and'].max()}, "
                f"OR crit max={merged['criticality_or'].max()})"
            )


if __name__ == "__main__":
    main()
