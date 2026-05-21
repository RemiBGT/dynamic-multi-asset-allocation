from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_market_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load the processed market dataset.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df.sort_index()
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns from price data.
    """
    return prices.pct_change()


def compute_drawdown(price_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Compute rolling drawdown over a given window.
    """
    rolling_max = price_series.rolling(window=window).max()
    drawdown = price_series / rolling_max - 1.0
    return drawdown


def build_regime_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build macro-financial features used for regime detection.

    Features are based on:
    - equity momentum and volatility
    - bond momentum
    - equity/bond correlation
    - market stress via VIX
    - rates, inflation and credit spreads
    """
    price_cols = [col for col in market_data.columns if col.startswith("px_")]
    prices = market_data[price_cols].rename(columns=lambda x: x.replace("px_", ""))
    returns = compute_returns(prices)

    features = pd.DataFrame(index=market_data.index)

    # Equity momentum and risk
    features["spy_ret_21d"] = prices["SPY"].pct_change(21)
    features["spy_ret_63d"] = prices["SPY"].pct_change(63)
    features["spy_vol_21d"] = returns["SPY"].rolling(21).std() * np.sqrt(252)
    features["spy_vol_63d"] = returns["SPY"].rolling(63).std() * np.sqrt(252)
    features["spy_drawdown_252d"] = compute_drawdown(prices["SPY"], window=252)

    # Fixed income momentum
    features["tlt_ret_21d"] = prices["TLT"].pct_change(21)
    features["ief_ret_21d"] = prices["IEF"].pct_change(21)
    features["hyg_ret_21d"] = prices["HYG"].pct_change(21)

    # Equity / bond diversification regime
    features["spy_tlt_corr_63d"] = returns["SPY"].rolling(63).corr(returns["TLT"])

    # Market stress
    features["vix_level"] = market_data["vix"]
    features["vix_change_21d"] = market_data["vix"].diff(21)

    # Rates regime
    features["us_10y_yield"] = market_data["us_10y_yield"]
    features["us_10y_change_21d"] = market_data["us_10y_yield"].diff(21)
    features["us_10y_2y_slope"] = market_data["us_10y_2y_slope"]
    features["us_10y_2y_slope_change_21d"] = market_data["us_10y_2y_slope"].diff(21)

    # Inflation regime
    features["breakeven_10y"] = market_data["breakeven_10y"]
    features["breakeven_10y_change_21d"] = market_data["breakeven_10y"].diff(21)
    features["real_yield_10y"] = market_data["real_yield_10y"]
    features["real_yield_10y_change_21d"] = market_data["real_yield_10y"].diff(21)

    # Credit stress
    features["baa_10y_spread"] = market_data["baa_10y_spread"]
    features["baa_10y_spread_change_21d"] = market_data["baa_10y_spread"].diff(21)
    features["aaa_10y_spread"] = market_data["aaa_10y_spread"]
    features["credit_quality_spread"] = (
        market_data["baa_10y_spread"] - market_data["aaa_10y_spread"]
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()

    return features


def build_investable_returns(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build clean daily returns for the investable ETF universe.
    """
    price_cols = [col for col in market_data.columns if col.startswith("px_")]
    prices = market_data[price_cols].rename(columns=lambda x: x.replace("px_", ""))
    returns = prices.pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def align_features_and_returns(
    features: pd.DataFrame,
    returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align features and returns on common dates.
    """
    common_index = features.index.intersection(returns.index)
    return features.loc[common_index], returns.loc[common_index]