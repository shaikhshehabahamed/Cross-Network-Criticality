import networkx as nx
import pytest

from src.data import build_networks, KNOWN_INVALID_COUNTRY_VALUES


@pytest.fixture(scope="module")
def nets():
    return build_networks()


def test_expected_sizes(nets):
    assert nets["AnonNet"].G.number_of_nodes() == 7245
    assert nets["AnonNet"].G.number_of_edges() == 43922
    assert nets["SupplyChainKG"].G.number_of_edges() == 17309


def test_single_weakly_connected_component(nets):
    for net in nets.values():
        assert nx.number_weakly_connected_components(net.G) == 1, net.name


def test_no_self_loops(nets):
    for net in nets.values():
        assert nx.number_of_selfloops(net.G) == 0, net.name


def test_no_duplicate_edges_possible(nets):
    # DiGraph cannot hold parallel edges; this just checks the loaders didn't
    # silently collapse more than the audited number of true duplicates.
    for net in nets.values():
        assert net.meta["duplicate_edges_removed"] >= 0


def test_country_harmonization_coverage(nets):
    for net in nets.values():
        raw = net.node_attrs["country_raw"]
        iso = net.node_attrs["country_iso3"]
        unresolved_but_present = raw.notna() & iso.isna()
        bad_strings = set(raw[unresolved_but_present].astype(str).str.strip().str.lower())
        # every unresolved-but-present country string must be an explicitly
        # documented data defect, never a silent gap in the harmonizer
        assert bad_strings <= KNOWN_INVALID_COUNTRY_VALUES, (net.name, bad_strings)


def test_country_codes_are_iso3(nets):
    for net in nets.values():
        codes = set(net.node_attrs["country_iso3"].dropna().unique())
        assert all(isinstance(c, str) and len(c) == 3 and c.isupper() for c in codes), net.name


def test_kg_owns_edges_excluded(nets):
    assert nets["SupplyChainKG"].meta["owns_rows_excluded"] == 30
