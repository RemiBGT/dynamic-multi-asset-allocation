from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.backtest import (  # noqa: E402
    backtest_benchmark_strategy,
    backtest_regime_strategy,
    combine_strategy_metrics,
    combine_strategy_returns,
    compute_drawdown_series,
    compute_performance_metrics,
    load_returns,
    load_walk_forward_regimes,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "backtests"
FIGURE_DIR = OUTPUT_DIR / "figures"


def plot_equity_curves(strategy_returns: pd.DataFrame) -> None:
    equity_curves = (1.0 + strategy_returns.fillna(0.0)).cumprod()

    plt.figure(figsize=(14, 7))

    for column in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[column], linewidth=1.2, label=column)

    plt.title("Strategy equity curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = FIGURE_DIR / "equity_curves_all_strategies.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_drawdowns(strategy_returns: pd.DataFrame) -> None:
    drawdowns = compute_drawdown_series(strategy_returns)

    plt.figure(figsize=(14, 7))

    for column in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[column], linewidth=1.2, label=column)

    plt.title("Strategy drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = FIGURE_DIR / "drawdowns_all_strategies.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_selected_equity_curves(strategy_returns: pd.DataFrame) -> None:
    selected_columns = [
        col
        for col in strategy_returns.columns
        if (
            "hmm_k3_hrp_tc5bps" in col
            or "hmm_k4_hrp_tc5bps" in col
            or "hmm_k3_risk_parity_tc5bps" in col
            or "hmm_k4_risk_parity_tc5bps" in col
            or "benchmark_static_6040_tc5bps" in col
        )
    ]

    if len(selected_columns) == 0:
        return

    equity_curves = (1.0 + strategy_returns[selected_columns].fillna(0.0)).cumprod()

    plt.figure(figsize=(14, 7))

    for column in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[column], linewidth=1.5, label=column)

    plt.title("Selected regime-aware strategies vs 60/40")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = FIGURE_DIR / "equity_curves_selected_strategies.png"
    plt.savefig(path, dpi=150)
    plt.close()


def build_common_period_outputs(all_results):
    """
    Recompute returns and metrics on a common period shared by all strategies.

    This avoids comparing a benchmark starting earlier with HMM strategies
    that only start after the walk-forward training period.
    """
    common_start = max(
        result.returns.dropna().index.min()
        for result in all_results
        if not result.returns.dropna().empty
    )

    common_returns_list = []
    common_metrics_rows = []

    for result in all_results:
        strategy_returns = result.returns.loc[common_start:].dropna()

        if strategy_returns.empty:
            continue

        aligned_turnover = result.turnover.reindex(strategy_returns.index).fillna(0.0)
        aligned_costs = result.costs.reindex(strategy_returns.index).fillna(0.0)

        metrics = compute_performance_metrics(
            returns=strategy_returns,
            turnover=aligned_turnover,
            costs=aligned_costs,
        )

        metrics["strategy"] = result.strategy_name
        metrics["n_components"] = result.metrics.get("n_components")
        metrics["allocation_method"] = result.metrics.get("allocation_method")
        metrics["transaction_cost_bps"] = result.metrics.get("transaction_cost_bps")

        common_metrics_rows.append(metrics)
        common_returns_list.append(strategy_returns.rename(result.strategy_name))

    common_returns = pd.concat(common_returns_list, axis=1).dropna(how="any")

    common_metrics = (
        pd.DataFrame(common_metrics_rows)
        .set_index("strategy")
        .sort_index()
    )

    return common_start, common_returns, common_metrics

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    print("Loading investable returns...")
    returns = load_returns(returns_path)

    all_results = []

    allocation_methods = [
        "equal_weight",
        "inverse_vol",
        "risk_parity",
        "hrp",
    ]

    transaction_costs_bps = [0.0, 5.0]

    hmm_components_to_test = [3, 4]

    for n_components in hmm_components_to_test:
        regimes_path = (
            ROOT_DIR
            / "reports"
            / "hmm"
            / "walk_forward"
            / f"walk_forward_results_k{n_components}.csv"
        )

        print(f"\nLoading walk-forward regimes K={n_components}...")
        regimes = load_walk_forward_regimes(regimes_path)

        for method in allocation_methods:
            for transaction_cost_bps in transaction_costs_bps:
                print(
                    f"Backtesting HMM K={n_components}, "
                    f"method={method}, "
                    f"tc={transaction_cost_bps:.0f}bps..."
                )

                result = backtest_regime_strategy(
                    returns=returns,
                    regimes=regimes,
                    n_components=n_components,
                    allocation_method=method,
                    transaction_cost_bps=transaction_cost_bps,
                    lookback_days=252,
                )

                result.returns.to_csv(
                    OUTPUT_DIR / f"{result.strategy_name}_returns.csv"
                )
                result.weights.to_csv(
                    OUTPUT_DIR / f"{result.strategy_name}_weights.csv"
                )

                all_results.append(result)

    benchmark_methods = [
        "static_6040",
        "equal_weight",
        "inverse_vol",
        "risk_parity",
        "hrp",
    ]

    for benchmark_method in benchmark_methods:
        for transaction_cost_bps in transaction_costs_bps:
            print(
                f"Backtesting benchmark={benchmark_method}, "
                f"tc={transaction_cost_bps:.0f}bps..."
            )

            result = backtest_benchmark_strategy(
                returns=returns,
                benchmark_method=benchmark_method,
                transaction_cost_bps=transaction_cost_bps,
                lookback_days=252,
            )

            result.returns.to_csv(
                OUTPUT_DIR / f"{result.strategy_name}_returns.csv"
            )
            result.weights.to_csv(
                OUTPUT_DIR / f"{result.strategy_name}_weights.csv"
            )

            all_results.append(result)

    print("\nCombining results...")

    # Full-period outputs: each strategy starts when it becomes active.
    strategy_returns_full = combine_strategy_returns(all_results)
    strategy_metrics_full = combine_strategy_metrics(all_results)

    strategy_returns_full.to_csv(OUTPUT_DIR / "strategy_returns_full_period.csv")
    strategy_metrics_full.to_csv(OUTPUT_DIR / "strategy_metrics_full_period.csv")

    # Common-period outputs: all strategies are evaluated on the same dates.
    common_start, strategy_returns_common, strategy_metrics_common = build_common_period_outputs(
        all_results
    )

    strategy_returns_common.to_csv(OUTPUT_DIR / "strategy_returns_common_period.csv")
    strategy_metrics_common.to_csv(OUTPUT_DIR / "strategy_metrics_common_period.csv")

    # Keep these names as the default final outputs for downstream analysis.
    strategy_returns_common.to_csv(OUTPUT_DIR / "strategy_returns.csv")
    strategy_metrics_common.to_csv(OUTPUT_DIR / "strategy_metrics.csv")

    # Final figures are now based on the common period.
    plot_equity_curves(strategy_returns_common)
    plot_drawdowns(strategy_returns_common)
    plot_selected_equity_curves(strategy_returns_common)

    print("\nBacktest complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Common evaluation period starts on: {common_start.date()}")

    display_columns = [
        "n_components",
        "allocation_method",
        "transaction_cost_bps",
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

    available_columns = [
        col for col in display_columns if col in strategy_metrics_common.columns
    ]

    print("\nTop strategies by Sharpe ratio on common period:")
    print(
        strategy_metrics_common[available_columns]
        .sort_values("sharpe_0rf", ascending=False)
        .head(15)
    )

    print("\nTop strategies by Calmar ratio on common period:")
    print(
        strategy_metrics_common[available_columns]
        .sort_values("calmar", ascending=False)
        .head(15)
    )

if __name__ == "__main__":
    main()