import networkx as nx
import pandas as pd

from src.data import Network, subgraph_excluding_missing_country


def test_excludes_only_missing_country_nodes_and_keeps_induced_edges():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
    node_attrs = pd.DataFrame(
        {"country_iso3": ["USA", None, "JPN", "GBR"]}, index=[0, 1, 2, 3]
    )
    net = Network(name="toy", G=G, node_attrs=node_attrs)
    excl = subgraph_excluding_missing_country(net)

    assert set(excl.G.nodes()) == {0, 2, 3}
    assert excl.meta["excluded_missing_country_nodes"] == 1
    # edges touching the dropped node (1) must be gone; edges among survivors kept
    assert (2, 3) in excl.G.edges()
    assert (3, 0) in excl.G.edges()
    assert not any(1 in e for e in excl.G.edges())
    assert excl.node_attrs["country_iso3"].isna().sum() == 0
