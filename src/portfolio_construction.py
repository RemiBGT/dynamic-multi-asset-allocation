from __future__ import annotations

from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


AllocationMethod = Literal["equal_weight", "inverse_vol", "risk_parity", "hrp"]


EQUITY_TICKERS = ["SPY", "QQQ", "IWM"]
BOND_TICKERS = ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG"]
ALL_TICKERS = EQUITY_TICKERS + BOND_TICKERS


REGIME_POLICIES: Dict[str, Dict[str, object]] = {
    "risk_on": {
        "equity_weight": 0.90,
        "bond_weight": 0.10,
        "equity_tickers": ["SPY", "QQQ", "IWM"],
        "bond_tickers": ["IEF", "TIP", "LQD", "HYG"],
    },
    "defensive_risk_off": {
        "equity_weight": 0.15,
        "bond_weight": 0.85,
        "equity_tickers": ["SPY"],
        "bond_tickers": ["SHY", "IEF", "TLT", "TIP"],
    },
    "inflation_rates_shock": {
        "equity_weight": 0.60,
        "bond_weight": 0.40,
        "equity_tickers": ["SPY", "IWM"],
        "bond_tickers": ["SHY", "TIP", "LQD"],
    },
    "recovery_or_normalization": {
        "equity_weight": 0.80,
        "bond_weight": 0.20,
        "equity_tickers": ["SPY", "QQQ", "IWM"],
        "bond_tickers": ["IEF", "TIP", "LQD", "HYG"],
    },
    "mixed_or_neutral": {
        "equity_weight": 0.60,
        "bond_weight": 0.40,
        "equity_tickers": ["SPY", "QQQ", "IWM"],
        "bond_tickers": ["SHY", "IEF", "TIP", "LQD"],
    },
}


def canonicalize_regime_label(regime_label: str) -> str:
    """
    Map unstable neutral labels into one generic neutral regime.
    """
    if regime_label.startswith("mixed_or_neutral"):
        return "mixed_or_neutral"

    if regime_label not in REGIME_POLICIES:
        return "mixed_or_neutral"

    return regime_label


def normalize_weights(weights: pd.Series) -> pd.Series:
    weights = weights.copy()
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = weights.clip(lower=0.0)

    total = weights.sum()

    if total <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)

    return weights / total


def equal_weight_weights(tickers: Iterable[str]) -> pd.Series:
    tickers = list(tickers)

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    return pd.Series(1.0 / len(tickers), index=tickers)


def inverse_vol_weights(
    returns_window: pd.DataFrame,
    min_volatility: float = 1e-6,
) -> pd.Series:
    """
    Simple risk-based allocation: weights proportional to inverse volatility.
    """
    volatility = returns_window.std() * np.sqrt(252)
    inv_vol = 1.0 / volatility.clip(lower=min_volatility)

    return normalize_weights(inv_vol)


def risk_parity_weights(
    returns_window: pd.DataFrame,
    max_iter: int = 1_000,
) -> pd.Series:
    """
    Long-only risk parity portfolio.

    Objective:
        equalize asset risk contributions.

    Risk contribution:
        RC_i = w_i * (Sigma w)_i / sqrt(w' Sigma w)
    """
    tickers = list(returns_window.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        return pd.Series(dtype=float)

    if n_assets == 1:
        return pd.Series(1.0, index=tickers)

    covariance = returns_window.cov().values * 252

    if not np.isfinite(covariance).all():
        return inverse_vol_weights(returns_window)

    def portfolio_volatility(weights: np.ndarray) -> float:
        variance = weights @ covariance @ weights
        return float(np.sqrt(max(variance, 1e-12)))

    def risk_contributions(weights: np.ndarray) -> np.ndarray:
        port_vol = portfolio_volatility(weights)
        marginal_risk = covariance @ weights
        return weights * marginal_risk / port_vol

    def objective(weights: np.ndarray) -> float:
        rc = risk_contributions(weights)
        target_rc = rc.mean()
        return float(((rc - target_rc) ** 2).sum())

    initial_weights = inverse_vol_weights(returns_window).reindex(tickers).values

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-12},
    )

    if not result.success:
        return inverse_vol_weights(returns_window)

    return normalize_weights(pd.Series(result.x, index=tickers))


def _correlation_distance(correlation: pd.DataFrame) -> pd.DataFrame:
    """
    Convert correlation matrix into a distance matrix for HRP clustering.

    d_ij = sqrt((1 - corr_ij) / 2)
    """
    clipped_corr = correlation.clip(lower=-1.0, upper=1.0)
    distance = np.sqrt((1.0 - clipped_corr) / 2.0)
    np.fill_diagonal(distance.values, 0.0)
    return distance


def _get_quasi_diag(linkage_matrix: np.ndarray, n_items: int) -> list[int]:
    """
    Sort clustered items by quasi-diagonalization of the linkage matrix.
    """
    linkage_matrix = linkage_matrix.astype(int)

    sort_index = pd.Series([linkage_matrix[-1, 0], linkage_matrix[-1, 1]])

    while sort_index.max() >= n_items:
        sort_index.index = range(0, sort_index.shape[0] * 2, 2)

        cluster_entries = sort_index[sort_index >= n_items]
        cluster_indices = cluster_entries.index
        linkage_indices = cluster_entries.values - n_items

        sort_index.loc[cluster_indices] = linkage_matrix[linkage_indices, 0]

        new_entries = pd.Series(
            linkage_matrix[linkage_indices, 1],
            index=cluster_indices + 1,
        )

        sort_index = pd.concat([sort_index, new_entries]).sort_index()
        sort_index.index = range(sort_index.shape[0])

    return sort_index.astype(int).tolist()


def _cluster_variance(
    covariance: pd.DataFrame,
    cluster_tickers: list[str],
) -> float:
    """
    Compute cluster variance using inverse-variance weights inside the cluster.
    """
    covariance_slice = covariance.loc[cluster_tickers, cluster_tickers]

    inv_diag = 1.0 / np.diag(covariance_slice).clip(min=1e-12)
    weights = inv_diag / inv_diag.sum()

    variance = weights @ covariance_slice.values @ weights

    return float(variance)


def _recursive_bisection(
    covariance: pd.DataFrame,
    ordered_tickers: list[str],
) -> pd.Series:
    """
    Recursive bisection step of Hierarchical Risk Parity.
    """
    weights = pd.Series(1.0, index=ordered_tickers)

    clusters = [ordered_tickers]

    while len(clusters) > 0:
        next_clusters = []

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2
            left_cluster = cluster[:split]
            right_cluster = cluster[split:]

            left_variance = _cluster_variance(covariance, left_cluster)
            right_variance = _cluster_variance(covariance, right_cluster)

            alpha = 1.0 - left_variance / (left_variance + right_variance)

            weights[left_cluster] *= alpha
            weights[right_cluster] *= 1.0 - alpha

            next_clusters.append(left_cluster)
            next_clusters.append(right_cluster)

        clusters = next_clusters

    return normalize_weights(weights)


def hrp_weights(returns_window: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity allocation.

    Steps:
    1. Estimate covariance and correlation.
    2. Convert correlation into distance.
    3. Apply hierarchical clustering.
    4. Quasi-diagonalize the covariance matrix.
    5. Allocate recursively by cluster variance.

    This is where the project uses clustering.
    """
    tickers = list(returns_window.columns)
    n_assets = len(tickers)

    if n_assets == 0:
        return pd.Series(dtype=float)

    if n_assets == 1:
        return pd.Series(1.0, index=tickers)

    covariance = returns_window.cov() * 252
    correlation = returns_window.corr()

    if not np.isfinite(covariance.values).all():
        return inverse_vol_weights(returns_window)

    if not np.isfinite(correlation.values).all():
        return inverse_vol_weights(returns_window)

    distance = _correlation_distance(correlation)
    condensed_distance = squareform(distance.values, checks=False)

    linkage_matrix = linkage(condensed_distance, method="single")
    sorted_indices = _get_quasi_diag(linkage_matrix, n_assets)
    ordered_tickers = [tickers[i] for i in sorted_indices]

    return _recursive_bisection(covariance, ordered_tickers).reindex(tickers)


def construct_bucket_weights(
    returns_window: pd.DataFrame,
    tickers: Iterable[str],
    method: AllocationMethod,
) -> pd.Series:
    """
    Construct weights inside one asset bucket: equity or fixed income.
    """
    tickers = [ticker for ticker in tickers if ticker in returns_window.columns]

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    bucket_returns = returns_window[tickers].dropna(how="any")

    if len(bucket_returns) < 60:
        return equal_weight_weights(tickers)

    if method == "equal_weight":
        return equal_weight_weights(tickers)

    if method == "inverse_vol":
        return inverse_vol_weights(bucket_returns)

    if method == "risk_parity":
        return risk_parity_weights(bucket_returns)

    if method == "hrp":
        return hrp_weights(bucket_returns)

    raise ValueError(f"Unknown allocation method: {method}")


def construct_regime_portfolio(
    regime_label: str,
    returns_window: pd.DataFrame,
    method: AllocationMethod,
    all_tickers: Iterable[str] = ALL_TICKERS,
) -> pd.Series:
    """
    Build the target portfolio weights for a given regime.

    The HMM gives the regime label.
    This function converts the regime label into:
    - an equity / bond split;
    - an intra-bucket allocation using equal weight, inverse vol, risk parity or HRP.
    """
    all_tickers = list(all_tickers)
    regime_label = canonicalize_regime_label(regime_label)

    policy = REGIME_POLICIES[regime_label]

    equity_weight = float(policy["equity_weight"])
    bond_weight = float(policy["bond_weight"])

    equity_tickers = list(policy["equity_tickers"])
    bond_tickers = list(policy["bond_tickers"])

    equity_bucket = construct_bucket_weights(
        returns_window=returns_window,
        tickers=equity_tickers,
        method=method,
    )

    bond_bucket = construct_bucket_weights(
        returns_window=returns_window,
        tickers=bond_tickers,
        method=method,
    )

    weights = pd.Series(0.0, index=all_tickers)

    if not equity_bucket.empty:
        weights.loc[equity_bucket.index] = equity_weight * equity_bucket

    if not bond_bucket.empty:
        weights.loc[bond_bucket.index] = bond_weight * bond_bucket

    return normalize_weights(weights)


def static_6040_weights() -> pd.Series:
    """
    Static 60/40 benchmark using SPY and IEF.
    """
    weights = pd.Series(0.0, index=ALL_TICKERS)
    weights["SPY"] = 0.60
    weights["IEF"] = 0.40
    return weights


def equal_weight_all_assets() -> pd.Series:
    """
    Static equal-weight benchmark across the full investable universe.
    """
    return equal_weight_weights(ALL_TICKERS).reindex(ALL_TICKERS).fillna(0.0)