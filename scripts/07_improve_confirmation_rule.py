from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.backtest import (  # noqa: E402
    BacktestResult,
    backtest_benchmark_strategy,
    backtest_regime_strategy,
    combine_strategy_metrics,
    combine_strategy_returns,
    compute_drawdown_series,
    compute_performance_metrics,
    load_returns,
    load_walk_forward_regimes,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "improvements" / "confirmation_rule"
FIGURE_DIR = OUTPUT_DIR / "figures"


def build_confirmed_regime_path(
    regimes: pd.DataFrame,
    confirmation_months: int,
) -> pd.DataFrame:
    """
    Build a confirmed regime path.

    A new regime becomes the confirmed regime only if it appears for
    confirmation_months consecutive decision dates.
    """
    valid = regimes[regimes["error"].fillna("") == ""].copy()
    valid = valid.dropna(subset=["current_regime_label"])

    candidate_labels = valid["current_regime_label"].astype(str)

    confirmed_labels = []

    current_confirmed = candidate_labels.iloc[0]
    pending_label = None
    pending_count = 0

    for candidate in candidate_labels:
        if candidate == current_confirmed:
            pending_label = None
            pending_count = 0
            confirmed_labels.append(current_confirmed)
            continue

        if candidate == pending_label:
            pending_count += 1
        else:
            pending_label = candidate
            pending_count = 1

        if pending_count >= confirmation_months:
            current_confirmed = candidate
            pending_label = None
            pending_count = 0

        confirmed_labels.append(current_confirmed)

    valid["raw_regime_label"] = valid["current_regime_label"]
    valid["confirmed_regime_label"] = confirmed_labels

    return valid

def build_event_driven_regime_schedule(
    confirmed_regimes: pd.DataFrame,
    max_holding_months: int | None,
) -> pd.DataFrame:
    """
    Keep only dates where the portfolio is actually rebalanced.

    Rebalance if:
    - the confirmed regime changes;
    - or max_holding_months is reached.
    """
    schedule = confirmed_regimes.copy()
    schedule["current_regime_label"] = schedule["confirmed_regime_label"]

    event_dates = []
    last_confirmed_label = None
    months_since_last_rebalance = 0

    for date, row in schedule.iterrows():
        confirmed_label = row["confirmed_regime_label"]

        if last_confirmed_label is None:
            event_dates.append(date)
            last_confirmed_label = confirmed_label
            months_since_last_rebalance = 0
            continue

        months_since_last_rebalance += 1

        regime_changed = confirmed_label != last_confirmed_label

        forced_rebalance = (
            max_holding_months is not None
            and months_since_last_rebalance >= max_holding_months
        )

        if regime_changed or forced_rebalance:
            event_dates.append(date)
            last_confirmed_label = confirmed_label
            months_since_last_rebalance = 0

    event_schedule = schedule.loc[event_dates].copy()

    return event_schedule

def rename_result(
    result: BacktestResult,
    new_name: str,
    confirmation_months: int,
    max_holding_months: int | None,
) -> BacktestResult:
    """
    Rename a backtest result and enrich its metrics with confirmation-rule parameters.
    """
    result.strategy_name = new_name
    result.returns = result.returns.rename(new_name)
    result.gross_returns = result.gross_returns.rename(new_name + "_gross")

    result.metrics["strategy"] = new_name
    result.metrics["confirmation_months"] = confirmation_months
    result.metrics["max_holding_months"] = (
        max_holding_months if max_holding_months is not None else -1
    )

    return result


def build_common_period_metrics(
    results: list[BacktestResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute returns and metrics on a common period shared by all strategies.
    """
    strategy_returns = combine_strategy_returns(results)

    common_start = max(
        strategy_returns[col].dropna().index.min()
        for col in strategy_returns.columns
    )

    common_returns = strategy_returns.loc[common_start:].dropna(how="any")

    rows = []

    for result in results:
        name = result.strategy_name

        if name not in common_returns.columns:
            continue

        aligned_turnover = result.turnover.reindex(common_returns.index).fillna(0.0)
        aligned_costs = result.costs.reindex(common_returns.index).fillna(0.0)

        metrics = compute_performance_metrics(
            returns=common_returns[name],
            turnover=aligned_turnover,
            costs=aligned_costs,
        )

        metrics["strategy"] = name
        metrics["n_components"] = result.metrics.get("n_components")
        metrics["allocation_method"] = result.metrics.get("allocation_method")
        metrics["transaction_cost_bps"] = result.metrics.get("transaction_cost_bps")
        metrics["confirmation_months"] = result.metrics.get("confirmation_months")
        metrics["max_holding_months"] = result.metrics.get("max_holding_months")

        rows.append(metrics)

    common_metrics = (
        pd.DataFrame(rows)
        .set_index("strategy")
        .sort_values("calmar", ascending=False)
    )

    return common_returns, common_metrics

def plot_equity_curves(
    returns: pd.DataFrame,
    filename: str,
    title: str,
) -> None:
    """
    Plot cumulative performance for a set of strategies.
    """
    equity_curves = (1.0 + returns.fillna(0.0)).cumprod()

    plt.figure(figsize=(14, 7))

    for column in equity_curves.columns:
        plt.plot(
            equity_curves.index,
            equity_curves[column],
            linewidth=1.4,
            label=column,
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=7)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()


def plot_drawdowns(
    returns: pd.DataFrame,
    filename: str,
    title: str,
) -> None:
    """
    Plot drawdowns for a set of strategies.
    """
    drawdowns = compute_drawdown_series(returns)

    plt.figure(figsize=(14, 7))

    for column in drawdowns.columns:
        plt.plot(
            drawdowns.index,
            drawdowns[column],
            linewidth=1.4,
            label=column,
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=7)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    print("Loading investable returns...")
    returns = load_returns(returns_path)

    all_results = []

    # Benchmarks for comparison
    benchmark_methods = [
        "static_6040",
        "equal_weight",
    ]

    for benchmark_method in benchmark_methods:
        print(f"Backtesting benchmark: {benchmark_method}")

        result = backtest_benchmark_strategy(
            returns=returns,
            benchmark_method=benchmark_method,
            transaction_cost_bps=5.0,
            lookback_days=252,
        )

        all_results.append(result)

    # Focus on the most promising HMM setup
    n_components = 3

    allocation_methods = [
        "hrp",
        "inverse_vol",
        "risk_parity",
    ]

    confirmation_configs = [
        {"confirmation_months": 1, "max_holding_months": 1},
        {"confirmation_months": 2, "max_holding_months": 3},
        {"confirmation_months": 2, "max_holding_months": 6},
        {"confirmation_months": 3, "max_holding_months": 6},
        {"confirmation_months": 3, "max_holding_months": 12},
    ]

    regimes_path = (
        ROOT_DIR
        / "reports"
        / "hmm"
        / "walk_forward"
        / f"walk_forward_results_k{n_components}.csv"
    )

    print(f"Loading HMM walk-forward regimes: {regimes_path}")
    raw_regimes = load_walk_forward_regimes(regimes_path)

    for config in confirmation_configs:
        confirmation_months = config["confirmation_months"]
        max_holding_months = config["max_holding_months"]

        confirmed_regimes = build_confirmed_regime_path(
            regimes=raw_regimes,
            confirmation_months=confirmation_months,
        )

        event_schedule = build_event_driven_regime_schedule(
            confirmed_regimes=confirmed_regimes,
            max_holding_months=max_holding_months,
        )

        print(
            f"\nConfirmation={confirmation_months}, "
            f"max_holding={max_holding_months}, "
            f"n_rebalance_dates={len(event_schedule)}"
        )

        event_schedule_path = (
            OUTPUT_DIR
            / f"event_schedule_k{n_components}"
            / f"_conf{confirmation_months}"
            / f"_max{max_holding_months}.csv"
        )
        event_schedule_path.parent.mkdir(parents=True, exist_ok=True)
        event_schedule.to_csv(event_schedule_path)

        for method in allocation_methods:
            print(
                f"Backtesting K={n_components}, method={method}, "
                f"confirmation={confirmation_months}, max={max_holding_months}"
            )

            result = backtest_regime_strategy(
                returns=returns,
                regimes=event_schedule,
                n_components=n_components,
                allocation_method=method,
                transaction_cost_bps=5.0,
                lookback_days=252,
            )

            new_name = (
                f"hmm_k{n_components}_{method}"
                f"_conf{confirmation_months}"
                f"_max{max_holding_months}"
                f"_tc5bps"
            )

            result = rename_result(
                result=result,
                new_name=new_name,
                confirmation_months=confirmation_months,
                max_holding_months=max_holding_months,
            )

            result.returns.to_csv(OUTPUT_DIR / f"{new_name}_returns.csv")
            result.weights.to_csv(OUTPUT_DIR / f"{new_name}_weights.csv")

            all_results.append(result)

    print("\nCombining results...")

    returns_full = combine_strategy_returns(all_results)
    metrics_full = combine_strategy_metrics(all_results)

    returns_common, metrics_common = build_common_period_metrics(all_results)

    returns_full.to_csv(OUTPUT_DIR / "confirmation_returns_full_period.csv")
    metrics_full.to_csv(OUTPUT_DIR / "confirmation_metrics_full_period.csv")

    returns_common.to_csv(OUTPUT_DIR / "confirmation_returns_common_period.csv")
    metrics_common.to_csv(OUTPUT_DIR / "confirmation_metrics_common_period.csv")

    selected_columns = [
        col
        for col in returns_common.columns
        if (
            col in [
                "benchmark_static_6040_tc5bps",
                "benchmark_equal_weight_tc5bps",
            ]
            or "hmm_k3_hrp_conf" in col
            or "hmm_k3_inverse_vol_conf" in col
        )
    ]

    selected_returns = returns_common[selected_columns].copy()

    plot_equity_curves(
        returns=selected_returns,
        filename="confirmation_equity_curves.png",
        title="Confirmed-regime strategies: equity curves",
    )

    plot_drawdowns(
        returns=selected_returns,
        filename="confirmation_drawdowns.png",
        title="Confirmed-regime strategies: drawdowns",
    )

    display_cols = [
        "n_components",
        "allocation_method",
        "transaction_cost_bps",
        "confirmation_months",
        "max_holding_months",
        "final_value",
        "cagr",
        "annual_volatility",
        "sharpe_0rf",
        "sortino_0rf",
        "max_drawdown",
        "calmar",
        "total_turnover",
        "total_transaction_costs",
    ]

    available_cols = [
        col for col in display_cols if col in metrics_common.columns
    ]

    print("\nTop strategies by Calmar on common period:")
    print(metrics_common[available_cols].head(20))

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()