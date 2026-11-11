"""Data loading, cleaning, and cross-network harmonization for AnonNet and SupplyChainKG.

Both networks are exposed as directed graphs where an edge u -> v means, under the
"as-given" convention, "u supplies v" (goods/relationship flow from source column to
target column, matching the raw column order in both source CSVs). This convention is
NOT independently confirmed by either dataset's documentation (see RESEARCH_DIRECTION.md
S1.1) and every downstream analysis must be re-run with edges reversed as a robustness
check -- see src/cascade.py and the sensitivity-analysis stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd
import pycountry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("data")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT

# Manual overrides for strings pycountry's fuzzy search cannot resolve or resolves
# ambiguously. Built by running _diagnose_unresolved() over both files' raw country
# columns and inspecting every miss by hand -- see tests/test_data.py::test_country_coverage.
COUNTRY_OVERRIDES = {
    "united states": "USA",
    "usa": "USA",
    "united states of america": "USA",
    "russia": "RUS",
    "russian federation": "RUS",
    "south korea": "KOR",
    "korea, republic of": "KOR",
    "north korea": "PRK",
    "taiwan": "TWN",
    "taiwan, province of china": "TWN",
    "vietnam": "VNM",
    "viet nam": "VNM",
    "iran": "IRN",
    "iran, islamic republic of": "IRN",
    "syria": "SYR",
    "laos": "LAO",
    "bolivia": "BOL",
    "venezuela": "VEN",
    "tanzania": "TZA",
    "moldova": "MDA",
    "brunei": "BRN",
    "ivory coast": "CIV",
    "cote d'ivoire": "CIV",
    "czech republic": "CZE",
    "czechia": "CZE",
    "macedonia": "MKD",
    "north macedonia": "MKD",
    "democratic republic of congo": "COD",
    "democratic republic of the congo": "COD",
    "congo, the democratic republic of the": "COD",
    "republic of congo": "COG",
    "congo": "COG",
    "united kingdom": "GBR",
    "uk": "GBR",
    "great britain": "GBR",
    "uae": "ARE",
    "united arab emirates": "ARE",
    "hong kong": "HKG",
    "macau": "MAC",
    "macao": "MAC",
    "palestine": "PSE",
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "cape verde": "CPV",
    "curacao": "CUW",
    "myanmar": "MMR",
    "burma": "MMR",
    "brasil": "BRA",
    'lao people"s democratic republic': "LAO",  # mojibake apostrophe in source CSV
    "lao people's democratic republic": "LAO",
    "libyan arab jamahiriya": "LBY",  # pre-2011 official name for Libya
    "philipines": "PHL",  # misspelling present in source CSV
    "turkey": "TUR",  # pycountry's default DB now indexes this under "Turkiye"
}

# Values found in supplychainKG.csv's country columns that are not countries at all --
# a documented upstream data defect, not a harmonization gap. Row 13703 (Zhejiang JULI
# Hi-tech Metals Co Ltd -> Baoji Titanium Industry Co Ltd) has "Steel" (an industry
# label) in the Customer Country field. Left unresolved (-> None -> counted as missing)
# rather than guessed at.
KNOWN_INVALID_COUNTRY_VALUES = {"steel"}


def _harmonize_one(raw: str, cache: dict) -> str | None:
    if raw is None or (isinstance(raw, float)):
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    if key in cache:
        return cache[key]
    if key in COUNTRY_OVERRIDES:
        cache[key] = COUNTRY_OVERRIDES[key]
        return cache[key]
    try:
        hit = pycountry.countries.lookup(raw)
        cache[key] = hit.alpha_3
        return cache[key]
    except LookupError:
        pass
    try:
        hits = pycountry.countries.search_fuzzy(raw)
        cache[key] = hits[0].alpha_3
        return cache[key]
    except LookupError:
        cache[key] = None
        return None


def harmonize_countries(raw_values: pd.Series) -> pd.Series:
    """Map free-text country names to ISO 3166-1 alpha-3 codes, shared across both
    networks. Returns None for unresolvable strings (logged, not silently dropped)."""
    cache: dict = {}
    out = raw_values.apply(lambda v: _harmonize_one(v, cache))
    unresolved_mask = out.isna() & raw_values.notna()
    unresolved = sorted(set(raw_values[unresolved_mask].astype(str)))
    unexpected = [v for v in unresolved if v.strip().lower() not in KNOWN_INVALID_COUNTRY_VALUES]
    if unexpected:
        log.warning("Could not harmonize %d non-null country strings: %s", len(unexpected), unexpected)
    known_bad = [v for v in unresolved if v.strip().lower() in KNOWN_INVALID_COUNTRY_VALUES]
    if known_bad:
        log.info("Treated %d known-invalid country value(s) as missing: %s", (raw_values.astype(str).isin(known_bad)).sum(), known_bad)
    return out


@dataclass
class Network:
    name: str
    G: nx.DiGraph
    node_attrs: pd.DataFrame  # indexed by node id: country_raw, country_iso3, plus extras
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        Gu = self.G.to_undirected()
        return {
            "name": self.name,
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "n_wcc": nx.number_weakly_connected_components(self.G),
            "density": nx.density(self.G),
            "self_loops_removed": self.meta.get("self_loops_removed", 0),
            "duplicate_edges_removed": self.meta.get("duplicate_edges_removed", 0),
            "country_missing_frac": float(self.node_attrs["country_iso3"].isna().mean()),
        }


def load_anonnet(data_dir: Path = DATA_DIR) -> Network:
    nodes = pd.read_csv(data_dir / "nodes_anon.csv")
    edges = pd.read_csv(data_dir / "edges_anon.csv")
    assert set(nodes.columns) == {"id", "location", "products"}
    assert set(edges.columns) == {"source", "target"}

    n_self_loops = int((edges["source"] == edges["target"]).sum())
    edges = edges[edges["source"] != edges["target"]].copy()
    n_dupes = int(edges.duplicated(subset=["source", "target"]).sum())
    edges = edges.drop_duplicates(subset=["source", "target"])

    G = nx.DiGraph()
    G.add_nodes_from(nodes["id"])
    G.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))

    node_attrs = nodes.set_index("id")
    node_attrs["country_iso3"] = harmonize_countries(node_attrs["location"])
    node_attrs = node_attrs.rename(columns={"location": "country_raw"})
    node_attrs["products_raw"] = node_attrs["products"]
    node_attrs["n_products"] = (
        node_attrs["products_raw"].dropna().str.split(",").apply(len)
    ).reindex(node_attrs.index)

    assert G.number_of_nodes() == 7245, f"expected 7245 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() == 43926 - n_self_loops, "unexpected edge count after cleaning"
    assert nx.number_weakly_connected_components(G) == 1, "AnonNet should be one component"

    return Network(
        name="AnonNet",
        G=G,
        node_attrs=node_attrs,
        meta={"self_loops_removed": n_self_loops, "duplicate_edges_removed": n_dupes},
    )


def load_supplychainkg(data_dir: Path = DATA_DIR) -> Network:
    kg = pd.read_csv(data_dir / "supplychainKG.csv")
    expected_cols = {
        "Supplier", "Supplier Industry", "Supplier Country",
        "Customer", "Customer Industry", "Customer Country",
        "Relationship1", "Relationship2", "Relationship3",
    }
    assert set(kg.columns) == expected_cols

    n_owns = int((kg["Relationship1"] == "owns").sum())
    supplies = kg[kg["Relationship1"] == "suppliesTo"].copy()
    n_self_loops = int((supplies["Supplier"] == supplies["Customer"]).sum())
    supplies = supplies[supplies["Supplier"] != supplies["Customer"]]
    n_dupes = int(supplies.duplicated(subset=["Supplier", "Customer"]).sum())
    supplies = supplies.drop_duplicates(subset=["Supplier", "Customer"])

    G = nx.DiGraph()
    G.add_edges_from(supplies[["Supplier", "Customer"]].itertuples(index=False, name=None))

    supplier_attrs = supplies[["Supplier", "Supplier Industry", "Supplier Country"]].rename(
        columns={"Supplier": "id", "Supplier Industry": "industry_raw", "Supplier Country": "country_raw"}
    )
    customer_attrs = supplies[["Customer", "Customer Industry", "Customer Country"]].rename(
        columns={"Customer": "id", "Customer Industry": "industry_raw", "Customer Country": "country_raw"}
    )
    node_attrs = (
        pd.concat([supplier_attrs, customer_attrs], ignore_index=True)
        .drop_duplicates(subset=["id"])
        .set_index("id")
    )
    node_attrs = node_attrs.reindex(list(G.nodes()))
    node_attrs["country_iso3"] = harmonize_countries(node_attrs["country_raw"])

    log.info("SupplyChainKG: excluded %d 'owns' relationship rows (kept suppliesTo only)", n_owns)

    assert nx.number_weakly_connected_components(G) == 1, "SupplyChainKG should be one component"

    return Network(
        name="SupplyChainKG",
        G=G,
        node_attrs=node_attrs,
        meta={
            "self_loops_removed": n_self_loops,
            "duplicate_edges_removed": n_dupes,
            "owns_rows_excluded": n_owns,
        },
    )


def subgraph_excluding_missing_country(net: Network) -> Network:
    """Listwise-exclusion variant for the missingness-handling sensitivity check (see
    RESEARCH_DIRECTION.md S3): drop every node with an unresolved country entirely
    (both from the graph and its attributes), rather than the default behavior
    elsewhere in the pipeline, which keeps such nodes and lets models learn from an
    explicit "country_missing" flag (src/features.py). Returns a fresh Network with the
    induced subgraph on the kept nodes -- ground truth cascades must be recomputed
    against this graph, not reused from the full-graph cache, since removing nodes
    changes downstream reachability for their neighbors.
    """
    keep = net.node_attrs.index[net.node_attrs["country_iso3"].notna()]
    G = net.G.subgraph(keep).copy()
    node_attrs = net.node_attrs.loc[list(G.nodes())]
    n_dropped = net.G.number_of_nodes() - G.number_of_nodes()
    return Network(
        name=net.name,
        G=G,
        node_attrs=node_attrs,
        meta={**net.meta, "excluded_missing_country_nodes": n_dropped},
    )


def build_networks(data_dir: Path = DATA_DIR) -> dict[str, Network]:
    return {"AnonNet": load_anonnet(data_dir), "SupplyChainKG": load_supplychainkg(data_dir)}


if __name__ == "__main__":
    nets = build_networks()
    for net in nets.values():
        for k, v in net.summary().items():
            print(f"{net.name:14s} {k}: {v}")
        print()
