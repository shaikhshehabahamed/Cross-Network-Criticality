"""Model-ready feature matrices for the Ridge/GNN cross-network transfer pipeline.

Two configurations, compared as an ablation (see scripts/run_ablations.py):
  topology_only    -- pure graph-structural features only. Maximally transferable
                       across networks; no vocabulary-alignment risk whatsoever.
  topology_country -- adds three numeric, within-network-normalized country-derived
                       features. Raw country IDENTITY (one-hot) is deliberately never
                       used as a transferable feature: AnonNet and SupplyChainKG were
                       collected independently and, while both country vocabularies are
                       harmonized to ISO3 (src/data.py), a model trained on network A's
                       one-hot country dimensions has no principled way to interpret
                       network B's -- see RESEARCH_DIRECTION.md S3. Free-text `products`
                       (AnonNet) vs. categorical `industry` (SupplyChainKG) have no
                       shared vocabulary at all and are excluded from the transfer
                       features for the same reason; they are used only in the
                       within-network descriptive/managerial analysis.

Log1p is applied to heavy-tailed count-like structural features before scaling: this
only affects optimizer conditioning, never the rank-based evaluation metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import Network

FEATURE_CONFIGS = ("topology_only", "topology_country")

_COUNT_LIKE_PREFIXES = ("in_degree", "out_degree", "betweenness")  # right-skewed -> log1p


def _log1p_skewed(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    for col in feats.columns:
        if col.startswith(_COUNT_LIKE_PREFIXES):
            feats[col] = np.log1p(feats[col])
    return feats


def _country_features(net: Network) -> pd.DataFrame:
    iso = net.node_attrs["country_iso3"].reindex(list(net.G.nodes()))
    freq = iso.value_counts()
    top5 = set(freq.head(5).index)
    log_freq = iso.map(freq).fillna(0).astype(float)
    log_freq = np.log1p(log_freq)
    z = (log_freq - log_freq.mean()) / (log_freq.std() + 1e-9)
    out = pd.DataFrame(
        {
            "country_is_top5": iso.isin(top5).astype(float),
            "country_log_freq_z": z,
            "country_missing": iso.isna().astype(float),
        },
        index=iso.index,
    )
    return out


def build_feature_matrix(net: Network, structural_feats: pd.DataFrame, config: str) -> pd.DataFrame:
    """Column order is sorted alphabetically unconditionally before returning -- this
    is what makes cross-network transfer safe regardless of which script produced the
    cached structural-feature parquet a given network reads from, or what order that
    parquet's columns happen to be in. Two independently-built caches with the same
    column SET but different column ORDER previously passed run_condition's set-based
    intuition but failed its list-equality assertion (see cache/features/AnonNet_excl_
    missing.parquet vs. the alphabetized cache/features/AnonNet.parquet from scripts/
    build_features.py) -- exactly the kind of silent train/test column misalignment
    that assertion exists to catch before it corrupts a model rather than after."""
    assert config in FEATURE_CONFIGS
    feats = _log1p_skewed(structural_feats.reindex(list(net.G.nodes())))
    feats = feats.fillna(feats.median())  # katz/betweenness occasionally missing for a metric, never a node
    if config == "topology_country":
        feats = feats.join(_country_features(net))
    feats = feats[sorted(feats.columns)]
    assert feats.isna().sum().sum() == 0, "unhandled NaNs in feature matrix"
    return feats


def fit_scaler_on_train(train_feats: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_feats.values)
    return scaler


def apply_scaler(feats: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    return pd.DataFrame(scaler.transform(feats.values), index=feats.index, columns=feats.columns)
