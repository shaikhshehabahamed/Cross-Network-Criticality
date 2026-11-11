"""Ground-truth cascading-impact simulation.

Two failure-propagation rules, both starting from a single failed seed node v0:

  OR  (zero redundancy / worst case): a node fails as soon as ANY one of its direct
      suppliers has failed. This is exactly forward reachability from v0.

  AND (full redundancy elsewhere / single-sourcing only): a node fails only once ALL
      of its direct suppliers have failed. A node with >1 supplier survives the loss of
      any strict subset of them. This is a standard bootstrap-percolation / threshold
      cascade.

AND-failures are always a subset of OR-failures for the same seed (a stricter condition
to satisfy) -- `criticality_AND(v) <= criticality_OR(v)` for every v is asserted as an
invariant on every run, real data included, not just in unit tests.

For each node v we record two scores:
  criticality(v)  = number of OTHER nodes that fail when v is removed  (impact if v fails)
  vulnerability(v) = number of OTHER nodes whose removal, alone, fails v (exposure of v)

criticality is a row-sum and vulnerability is the matching column-sum of the same
(v -> failed-set) relation, so sum(criticality) == sum(vulnerability) exactly for both
models -- also asserted on every run.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _run_cascades(indptr, indices, in_degree, mode_and: bool):
    n = indptr.shape[0] - 1
    criticality = np.zeros(n, dtype=np.int64)
    vulnerability = np.zeros(n, dtype=np.int64)
    failed = np.zeros(n, dtype=np.bool_)
    remaining = np.zeros(n, dtype=np.int64)
    stack = np.zeros(n, dtype=np.int64)
    failed_order = np.zeros(n, dtype=np.int64)

    for start in range(n):
        failed[:] = False
        if mode_and:
            remaining[:] = in_degree
        failed[start] = True
        failed_order[0] = start
        n_failed = 1
        stack[0] = start
        top = 1
        while top > 0:
            top -= 1
            w = stack[top]
            for e in range(indptr[w], indptr[w + 1]):
                x = indices[e]
                if failed[x]:
                    continue
                if mode_and:
                    remaining[x] -= 1
                    should_fail = remaining[x] == 0
                else:
                    should_fail = True
                if should_fail:
                    failed[x] = True
                    failed_order[n_failed] = x
                    n_failed += 1
                    stack[top] = x
                    top += 1
        criticality[start] = n_failed - 1
        for i in range(n_failed):
            vulnerability[failed_order[i]] += 1
        vulnerability[start] -= 1

    return criticality, vulnerability


@dataclass
class CascadeResult:
    mode: str          # "AND" or "OR"
    direction: str      # "as_given" or "reversed"
    scores: pd.DataFrame  # indexed by node id: columns criticality, vulnerability


def _to_csr(G: nx.DiGraph, nodes: list):
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    src = np.empty(G.number_of_edges(), dtype=np.int64)
    dst = np.empty(G.number_of_edges(), dtype=np.int64)
    for e, (u, v) in enumerate(G.edges()):
        src[e] = idx[u]
        dst[e] = idx[v]
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    counts = np.bincount(src, minlength=n)
    indptr[1:] = np.cumsum(counts)
    in_degree = np.bincount(dst, minlength=n).astype(np.int64)
    return indptr, dst.astype(np.int64), in_degree


def run_cascade(G: nx.DiGraph, mode: str = "AND", direction: str = "as_given") -> CascadeResult:
    """mode: 'AND' or 'OR'. direction: 'as_given' (edges as stored) or 'reversed'."""
    assert mode in ("AND", "OR")
    assert direction in ("as_given", "reversed")
    H = G if direction == "as_given" else G.reverse(copy=False)
    nodes = list(G.nodes())  # fix node order regardless of direction, for aligned output
    indptr, indices, in_degree = _to_csr(H, nodes)
    criticality, vulnerability = _run_cascades(indptr, indices, in_degree, mode == "AND")

    total_c, total_v = int(criticality.sum()), int(vulnerability.sum())
    assert total_c == total_v, f"row/column sum mismatch: {total_c} != {total_v} ({mode}, {direction})"
    assert criticality.min() >= 0 and vulnerability.min() >= 0

    scores = pd.DataFrame(
        {"criticality": criticality, "vulnerability": vulnerability}, index=pd.Index(nodes, name="node")
    )
    return CascadeResult(mode=mode, direction=direction, scores=scores)


if __name__ == "__main__":
    import time

    from src.data import build_networks

    nets = build_networks()
    for net in nets.values():
        for direction in ("as_given", "reversed"):
            or_res = run_cascade(net.G, "OR", direction)
            t0 = time.time()
            and_res = run_cascade(net.G, "AND", direction)
            dt = time.time() - t0
            assert (and_res.scores["criticality"] <= or_res.scores["criticality"]).all(), "AND<=OR violated"
            assert (and_res.scores["vulnerability"] <= or_res.scores["vulnerability"]).all(), "AND<=OR violated (vuln)"
            print(
                f"{net.name:14s} dir={direction:9s} AND max_crit={and_res.scores['criticality'].max():5d} "
                f"OR max_crit={or_res.scores['criticality'].max():5d} (AND cascade took {dt:.2f}s)"
            )
