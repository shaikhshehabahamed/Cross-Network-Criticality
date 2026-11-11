"""Structural (no-training) predictors: standard centrality/connectivity metrics.

PageRank and Katz centrality are genuinely direction-dependent (both are defined via a
recursive/random-walk relation over IN-edges specifically), so both are computed on G
as given ("_fwd") and on G.reverse() ("_rev") and kept as separate features, letting
the modeling stage learn which direction best matches criticality vs. vulnerability
rather than that choice being hand-picked upfront.

Degree and betweenness are each computed exactly ONCE, not per direction, because a
"_rev" version would be an exact duplicate of an already-included column:
  - in_degree(G.reverse()) == out_degree(G) and vice versa, by definition of reversal.
  - directed betweenness centrality is provably invariant under full graph reversal:
    it sums, over every ORDERED pair (s, t), whether v lies on a shortest s->t path;
    reversing every edge turns each shortest s->t path into a shortest t->s path, so
    reversal is just a relabeling of the same sum over ordered pairs.
Both redundancies were caught empirically, not just derived on paper: the first via a
scikit-learn ill-conditioned-matrix warning during smoke testing, the second via the
general pairwise-correlation invariant test test_no_duplicate_or_perfectly_correlated_
columns -- see tests/test_predictors.py. Including either duplicate costs nothing
visibly (Ridge with L2 regularization tolerates collinearity) but is still wrong: it
misleadingly inflates the apparent feature count and, at low alpha, destabilizes the
linear solve.

Every metric is wrapped defensively -- numerical non-convergence on an irregular
real-world graph (e.g. eigenvector-family centralities) must not crash the pipeline; a
failed metric is logged and dropped, not silently zero-filled.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import ArpackNoConvergence, eigs

log = logging.getLogger("predictors")

APPROX_BETWEENNESS_K = 1000  # sampled sources; see build_structural_features docstring


def _safe_katz(G: nx.DiGraph) -> dict | None:
    try:
        A = nx.adjacency_matrix(G, weight=None).astype(float)
        if A.shape[0] < 3:
            return None
        eigval = eigs(A.T, k=1, which="LM", return_eigenvectors=False, maxiter=5000)
        spectral_radius = abs(eigval[0])
        if spectral_radius <= 0:
            return None
        alpha = 0.9 / spectral_radius
        return nx.katz_centrality_numpy(G, alpha=alpha, weight=None)
    except (ArpackNoConvergence, np.linalg.LinAlgError, nx.NetworkXError) as e:
        log.warning("katz centrality failed (%s); dropping this feature", e)
        return None


def build_structural_features(G: nx.DiGraph, seed: int = 0) -> pd.DataFrame:
    """Direction-invariant metrics computed once (in_degree, out_degree, betweenness,
    k_core) plus the genuinely direction-dependent pair (pagerank, katz) computed on G
    and on G.reverse().

    Betweenness is approximated via `k`-sampled sources (Brandes with pivot sampling,
    the standard approach for graphs of this size where exact betweenness over all
    ~7,000 sources is unnecessarily slow); k=1000 is a stability/runtime compromise.
    """
    nodes = list(G.nodes())
    Gr = G.reverse(copy=False)
    feats = pd.DataFrame(index=pd.Index(nodes, name="node"))

    feats["in_degree"] = pd.Series(dict(G.in_degree()))
    feats["out_degree"] = pd.Series(dict(G.out_degree()))

    try:
        feats["betweenness"] = pd.Series(
            nx.betweenness_centrality(G, k=min(APPROX_BETWEENNESS_K, G.number_of_nodes()), seed=seed, normalized=True)
        )
    except Exception as e:
        log.warning("betweenness failed (%s); dropping this feature", e)

    feats["pagerank_fwd"] = pd.Series(nx.pagerank(G, alpha=0.85))
    feats["pagerank_rev"] = pd.Series(nx.pagerank(Gr, alpha=0.85))

    katz_fwd = _safe_katz(G)
    if katz_fwd is not None:
        feats["katz_fwd"] = pd.Series(katz_fwd)
    katz_rev = _safe_katz(Gr)
    if katz_rev is not None:
        feats["katz_rev"] = pd.Series(katz_rev)

    core = nx.core_number(nx.Graph(G))  # k-core is only standard for simple undirected graphs
    feats["k_core"] = pd.Series(core)

    feats = feats.reindex(nodes)
    assert feats.index.equals(pd.Index(nodes, name="node"))
    return feats


if __name__ == "__main__":
    from src.data import build_networks

    nets = build_networks()
    for net in nets.values():
        feats = build_structural_features(net.G, seed=0)
        print(f"{net.name}: {feats.shape[1]} structural features, {feats.shape[0]} nodes")
        print(feats.describe().T[["mean", "std", "min", "max"]])
        print()
