from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd

from src.portfolio_construction import (
    ALL_TICKERS,
    AllocationMethod,
    construct_regime_portfolio,
    equal_weight_all_assets,
    hrp_weights,
    inverse_vol_weights,
    risk_parity_weights,
    static_6040_weights,
)


BenchmarkMethod = Literal[
    "static_6040",
    "equal_weight",
    "inverse_vol",
    "risk_parity",
    "hrp",
]


@dataclass
class BacktestResult:
    strategy_name: str
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: Dict[str, float]


def load_returns(path: str | Path) -> pd.DataFrame:
    returns = pd.read_csv(path, index_col="date", parse_dates=True)
    returns = returns.sort_index()
    return returns


def load_walk_forward_regimes(path: str | Path) -> pd.DataFrame:
    regimes = pd.read_csv(path, index_col="date", parse_dates=True)
    regimes = regimes.sort_index()
    return regimes


def get_next_trading_date(
    trading_dates: pd.DatetimeIndex,
    date: pd.Timestamp,
) -> pd.Timestamp | None:
    candidates = trading_dates[trading_dates > date]

    if len(candidates) == 0:
        return None

    return candidates[0]


def compute_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series,
) -> float:
    current_weights = current_weights.reindex(previous_weights.index).fillna(0.0)
    previous_weights = previous_weights.reindex(current_weights.index).fillna(0.0)

    return float(np.abs(current_weights - previous_weights).sum())


def compute_performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    returns = returns.dropna()

    if returns.empty:
        return {}

    equity_curve = (1.0 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0

    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    final_value = float(equity_curve.iloc[-1])

    if n_years > 0:
        cagr = final_value ** (1.0 / n_years) - 1.0
    else:
        cagr = np.nan

    annual_return = returns.mean() * periods_per_year
    annual_volatility = returns.std() * np.sqrt(periods_per_year)

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(periods_per_year)

    sortino = (
        annual_return / downside_volatility
        if downside_volatility > 0
        else np.nan
    )

    max_drawdown = float(drawdown.min())

    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    var_95 = float(returns.quantile(0.05))
    cvar_95 = float(returns[returns <= var_95].mean())

    positive_days = (returns > 0).mean()

    metrics = {
        "start_date": returns.index.min(),
        "end_date": returns.index.max(),
        "n_days": n_periods,
        "final_value": final_value,
        "cagr": cagr,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_0rf": sharpe,
        "sortino_0rf": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "positive_day_ratio": positive_days,
    }

    if turnover is not None and not turnover.empty:
        metrics["total_turnover"] = float(turnover.sum())
        metrics["average_rebalance_turnover"] = float(turnover[turnover > 0].mean())
        metrics["n_rebalances"] = int((turnover > 0).sum())

    if costs is not None and not costs.empty:
        metrics["total_transaction_costs"] = float(costs.sum())
        metrics["average_transaction_cost"] = float(costs[costs > 0].mean())

    return metrics


def build_static_benchmark_weights(
    method: BenchmarkMethod,
    returns_window: pd.DataFrame,
) -> pd.Series:
    if method == "static_6040":
        return static_6040_weights()

    if method == "equal_weight":
        return equal_weight_all_assets()

    if method == "inverse_vol":
        return inverse_vol_weights(returns_window[ALL_TICKERS])

    if method == "risk_parity":
        return risk_parity_weights(returns_window[ALL_TICKERS])

    if method == "hrp":
        return hrp_weights(returns_window[ALL_TICKERS])

    raise ValueError(f"Unknown benchmark method: {method}")


def backtest_regime_strategy(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    n_components: int,
    allocation_method: AllocationMethod,
    transaction_cost_bps: float = 0.0,
    lookback_days: int = 252,
    all_tickers: Iterable[str] = ALL_TICKERS,
) -> BacktestResult:
    """
    Backtest a regime-aware allocation strategy.

    At each walk-forward HMM decision date:
    - use the current regime label;
    - estimate intra-bucket weights using only past returns;
    - trade at the close of the decision date;
    - apply new weights from the next trading day onward.
    """
    all_tickers = list(all_tickers)
    returns = returns[all_tickers].dropna(how="any").copy()
    regimes = regimes.copy()

    valid_regimes = regimes[regimes["error"].fillna("") == ""].copy()
    valid_regimes = valid_regimes.dropna(subset=["current_regime_label"])

    trading_dates = returns.index

    weights = pd.DataFrame(0.0, index=returns.index, columns=all_tickers)
    turnover = pd.Series(0.0, index=returns.index, name="turnover")
    costs = pd.Series(0.0, index=returns.index, name="transaction_costs")

    previous_weights = pd.Series(0.0, index=all_tickers)

    decision_dates = valid_regimes.index

    for i, decision_date in enumerate(decision_dates):
        next_trading_date = get_next_trading_date(trading_dates, decision_date)

        if next_trading_date is None:
            continue

        if i + 1 < len(decision_dates):
            next_decision_date = decision_dates[i + 1]
            holding_dates = trading_dates[
                (trading_dates >= next_trading_date)
                & (trading_dates <= next_decision_date)
            ]
        else:
            holding_dates = trading_dates[trading_dates >= next_trading_date]

        if len(holding_dates) == 0:
            continue

        historical_window = returns.loc[:decision_date].tail(lookback_days)

        if len(historical_window) < 60:
            continue

        regime_label = str(valid_regimes.loc[decision_date, "current_regime_label"])

        target_weights = construct_regime_portfolio(
            regime_label=regime_label,
            returns_window=historical_window,
            method=allocation_method,
            all_tickers=all_tickers,
        )

        target_weights = target_weights.reindex(all_tickers).fillna(0.0)

        rebalance_turnover = compute_turnover(
            current_weights=target_weights,
            previous_weights=previous_weights,
        )

        transaction_cost = transaction_cost_bps / 10_000.0 * rebalance_turnover

        turnover.loc[next_trading_date] = rebalance_turnover
        costs.loc[next_trading_date] = transaction_cost

        weights.loc[holding_dates, :] = target_weights.values

        previous_weights = target_weights.copy()

    gross_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    net_returns = gross_returns - costs

    active_mask = weights.abs().sum(axis=1) > 0
    net_returns = net_returns.loc[active_mask]
    gross_returns = gross_returns.loc[active_mask]
    costs = costs.loc[active_mask]
    turnover = turnover.loc[active_mask]
    weights = weights.loc[active_mask]

    strategy_name = (
        f"hmm_k{n_components}_{allocation_method}_tc{transaction_cost_bps:.0f}bps"
    )

    metrics = compute_performance_metrics(
        returns=net_returns,
        turnover=turnover,
        costs=costs,
    )

    metrics["strategy"] = strategy_name
    metrics["n_components"] = n_components
    metrics["allocation_method"] = allocation_method
    metrics["transaction_cost_bps"] = transaction_cost_bps

    return BacktestResult(
        strategy_name=strategy_name,
        returns=net_returns.rename(strategy_name),
        gross_returns=gross_returns.rename(strategy_name + "_gross"),
        costs=costs,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
    )


def backtest_benchmark_strategy(
    returns: pd.DataFrame,
    benchmark_method: BenchmarkMethod,
    transaction_cost_bps: float = 0.0,
    lookback_days: int = 252,
    rebalance_frequency: str = "M",
    all_tickers: Iterable[str] = ALL_TICKERS,
) -> BacktestResult:
    """
    Backtest benchmark strategies.

    Benchmarks:
    - static_6040: constant SPY/IEF 60/40, rebalanced monthly;
    - equal_weight: equal-weight across all assets, rebalanced monthly;
    - inverse_vol / risk_parity / hrp: rolling risk-based allocation
      without regime conditioning.
    """
    all_tickers = list(all_tickers)
    returns = returns[all_tickers].dropna(how="any").copy()

    trading_dates = returns.index

    month_end_dates = (
        returns.index.to_series()
        .groupby(returns.index.to_period(rebalance_frequency))
        .max()
    )

    weights = pd.DataFrame(0.0, index=returns.index, columns=all_tickers)
    turnover = pd.Series(0.0, index=returns.index, name="turnover")
    costs = pd.Series(0.0, index=returns.index, name="transaction_costs")

    previous_weights = pd.Series(0.0, index=all_tickers)

    decision_dates = pd.DatetimeIndex(month_end_dates.values)

    for i, decision_date in enumerate(decision_dates):
        next_trading_date = get_next_trading_date(trading_dates, decision_date)

        if next_trading_date is None:
            continue

        if i + 1 < len(decision_dates):
            next_decision_date = decision_dates[i + 1]
            holding_dates = trading_dates[
                (trading_dates >= next_trading_date)
                & (trading_dates <= next_decision_date)
            ]
        else:
            holding_dates = trading_dates[trading_dates >= next_trading_date]

        if len(holding_dates) == 0:
            continue

        historical_window = returns.loc[:decision_date].tail(lookback_days)

        if len(historical_window) < 60:
            continue

        target_weights = build_static_benchmark_weights(
            method=benchmark_method,
            returns_window=historical_window,
        )

        target_weights = target_weights.reindex(all_tickers).fillna(0.0)

        rebalance_turnover = compute_turnover(
            current_weights=target_weights,
            previous_weights=previous_weights,
        )

        transaction_cost = transaction_cost_bps / 10_000.0 * rebalance_turnover

        turnover.loc[next_trading_date] = rebalance_turnover
        costs.loc[next_trading_date] = transaction_cost
        weights.loc[holding_dates, :] = target_weights.values

        previous_weights = target_weights.copy()

    gross_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    net_returns = gross_returns - costs

    active_mask = weights.abs().sum(axis=1) > 0
    net_returns = net_returns.loc[active_mask]
    gross_returns = gross_returns.loc[active_mask]
    costs = costs.loc[active_mask]
    turnover = turnover.loc[active_mask]
    weights = weights.loc[active_mask]

    strategy_name = f"benchmark_{benchmark_method}_tc{transaction_cost_bps:.0f}bps"

    metrics = compute_performance_metrics(
        returns=net_returns,
        turnover=turnover,
        costs=costs,
    )

    metrics["strategy"] = strategy_name
    metrics["n_components"] = np.nan
    metrics["allocation_method"] = benchmark_method
    metrics["transaction_cost_bps"] = transaction_cost_bps

    return BacktestResult(
        strategy_name=strategy_name,
        returns=net_returns.rename(strategy_name),
        gross_returns=gross_returns.rename(strategy_name + "_gross"),
        costs=costs,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
    )


def combine_strategy_returns(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.concat([result.returns for result in results], axis=1).dropna(how="all")


def combine_strategy_metrics(results: list[BacktestResult]) -> pd.DataFrame:
    metrics = pd.DataFrame([result.metrics for result in results])
    return metrics.set_index("strategy").sort_index()


def compute_drawdown_series(returns: pd.DataFrame) -> pd.DataFrame:
    equity_curves = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = equity_curves.cummax()
    return equity_curves / running_max - 1.0

def backtest_weight_schedule(
    strategy_name: str,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> BacktestResult:
    """
    Backtest a strategy from a precomputed daily weight schedule.

    The function:
    - aligns weights and returns;
    - computes daily turnover from weight changes;
    - applies proportional transaction costs;
    - computes net returns and performance metrics.
    """
    common_columns = [col for col in weights.columns if col in returns.columns]

    returns = returns[common_columns].copy()
    weights = weights[common_columns].copy()

    weights = weights.reindex(returns.index).ffill().fillna(0.0)

    active_mask = weights.abs().sum(axis=1) > 0
    returns = returns.loc[active_mask]
    weights = weights.loc[active_mask]

    if returns.empty:
        raise ValueError(f"No active period found for strategy {strategy_name}.")

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    turnover.name = "turnover"

    costs = transaction_cost_bps / 10_000.0 * turnover
    costs.name = "transaction_costs"

    gross_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    gross_returns.name = strategy_name + "_gross"

    net_returns = gross_returns - costs
    net_returns.name = strategy_name

    metrics = compute_performance_metrics(
        returns=net_returns,
        turnover=turnover,
        costs=costs,
    )

    metrics["strategy"] = strategy_name
    metrics["n_components"] = np.nan
    metrics["allocation_method"] = "overlay"
    metrics["transaction_cost_bps"] = transaction_cost_bps

    return BacktestResult(
        strategy_name=strategy_name,
        returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
    )