import networkx as nx
import pytest

from src.cascade import run_cascade


def crit(res, node):
    return int(res.scores.loc[node, "criticality"])


def vuln(res, node):
    return int(res.scores.loc[node, "vulnerability"])


# ---------------------------------------------------------------------------
# Chain: 0 -> 1 -> 2 -> 3 -> 4. Every node has in-degree <= 1, so AND == OR here.
# ---------------------------------------------------------------------------
def test_chain_or_and_agree_and_values():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
    for mode in ("OR", "AND"):
        res = run_cascade(G, mode, "as_given")
        assert crit(res, 0) == 4
        assert crit(res, 2) == 2
        assert crit(res, 4) == 0
        assert vuln(res, 4) == 4
        assert vuln(res, 0) == 0
        assert vuln(res, 2) == 2  # only 0 and 1 upstream


def test_chain_reversed_direction_flips_roles():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
    res = run_cascade(G, "OR", "reversed")
    # reversed: node 4 now looks like the "root" whose failure cascades to 0..3
    assert crit(res, 4) == 4
    assert crit(res, 0) == 0


# ---------------------------------------------------------------------------
# Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3. Node 3 has two suppliers (1 and 2) --
# this is the key AND vs OR divergence case (redundancy protects node 3).
# ---------------------------------------------------------------------------
@pytest.fixture
def diamond():
    return nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])


def test_diamond_removing_root_fails_everything_both_models(diamond):
    for mode in ("OR", "AND"):
        res = run_cascade(diamond, mode, "as_given")
        assert crit(res, 0) == 3, mode  # 0 takes out 1, 2, and (transitively) 3


def test_diamond_or_overestimates_relative_to_and(diamond):
    or_res = run_cascade(diamond, "OR", "as_given")
    and_res = run_cascade(diamond, "AND", "as_given")
    # Removing 1 alone: OR says 3 fails too (node 3 loses *a* supplier).
    assert crit(or_res, 1) == 1
    # AND says 3 survives: it still has supplier 2 intact.
    assert crit(and_res, 1) == 0
    # symmetric for node 2
    assert crit(or_res, 2) == 1
    assert crit(and_res, 2) == 0


def test_diamond_vulnerability_reflects_redundancy(diamond):
    or_res = run_cascade(diamond, "OR", "as_given")
    and_res = run_cascade(diamond, "AND", "as_given")
    # OR: node 3 is "vulnerable" to any of 0, 1, 2 failing alone -> 3
    assert vuln(or_res, 3) == 3
    # AND: only node 0's removal alone can take out node 3 (it kills both suppliers)
    assert vuln(and_res, 3) == 1


# ---------------------------------------------------------------------------
# Star: hub 0 supplies 4 independent leaves with no further downstream edges.
# ---------------------------------------------------------------------------
def test_star_hub_and_leaves():
    G = nx.DiGraph([(0, 1), (0, 2), (0, 3), (0, 4)])
    for mode in ("OR", "AND"):
        res = run_cascade(G, mode, "as_given")
        assert crit(res, 0) == 4
        for leaf in (1, 2, 3, 4):
            assert crit(res, leaf) == 0
            assert vuln(res, leaf) == 1
        assert vuln(res, 0) == 0


# ---------------------------------------------------------------------------
# Cycle with a pendant: 0 -> 1 -> 2 -> 0, and 2 -> 3. Every node in-degree == 1,
# so AND == OR; this case exists to confirm cycles terminate and are handled.
# ---------------------------------------------------------------------------
def test_cycle_with_pendant_terminates_and_is_correct():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)])
    for mode in ("OR", "AND"):
        res = run_cascade(G, mode, "as_given")
        assert crit(res, 0) == 3  # 1, 2, 3 all fail
        assert crit(res, 3) == 0  # pendant has no downstream


# ---------------------------------------------------------------------------
# Isolated node: no edges at all.
# ---------------------------------------------------------------------------
def test_isolated_node():
    G = nx.DiGraph()
    G.add_node(0)
    res = run_cascade(G, "OR", "as_given")
    assert crit(res, 0) == 0
    assert vuln(res, 0) == 0


# ---------------------------------------------------------------------------
# Cross-checks against independent networkx primitives, and the general
# invariants that must hold on ANY graph (including the real datasets).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "edges",
    [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        [(0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 1), (1, 2), (2, 0), (2, 3)],
        [(0, 1), (1, 2), (1, 3), (3, 4), (4, 1), (2, 5)],  # bigger graph w/ nested cycle
    ],
)
def test_or_model_matches_networkx_descendants_and_ancestors(edges):
    G = nx.DiGraph(edges)
    res = run_cascade(G, "OR", "as_given")
    for node in G.nodes():
        assert crit(res, node) == len(nx.descendants(G, node)), node
        assert vuln(res, node) == len(nx.ancestors(G, node)), node


@pytest.mark.parametrize(
    "edges",
    [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        [(0, 1), (1, 2), (1, 3), (3, 4), (4, 1), (2, 5)],
    ],
)
@pytest.mark.parametrize("direction", ["as_given", "reversed"])
def test_and_never_exceeds_or_and_sums_match(edges, direction):
    G = nx.DiGraph(edges)
    or_res = run_cascade(G, "OR", direction)
    and_res = run_cascade(G, "AND", direction)
    merged = or_res.scores.join(and_res.scores, lsuffix="_or", rsuffix="_and")
    assert (merged["criticality_and"] <= merged["criticality_or"]).all()
    assert (merged["vulnerability_and"] <= merged["vulnerability_or"]).all()
    assert or_res.scores["criticality"].sum() == or_res.scores["vulnerability"].sum()
    assert and_res.scores["criticality"].sum() == and_res.scores["vulnerability"].sum()
