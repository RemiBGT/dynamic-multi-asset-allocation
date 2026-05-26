from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.backtest import (  # noqa: E402
    backtest_weight_schedule,
    compute_drawdown_series,
    compute_performance_metrics,
    load_returns,
)


BACKTEST_DIR = ROOT_DIR / "reports" / "backtests"
OUTPUT_DIR = ROOT_DIR / "reports" / "improvements"
FIGURE_DIR = OUTPUT_DIR / "figures"


def load_strategy_returns() -> pd.DataFrame:
    path = BACKTEST_DIR / "strategy_returns.csv"
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()


def load_strategy_weights(strategy_name: str) -> pd.DataFrame:
    path = BACKTEST_DIR / f"{strategy_name}_weights.csv"
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()


def first_valid_date(series: pd.Series) -> pd.Timestamp:
    valid = series.dropna()
    if valid.empty:
        raise ValueError(f"No valid data found for {series.name}.")
    return valid.index.min()


def compute_common_period_metrics(
    strategy_returns: pd.DataFrame,
    selected_strategies: list[str],
) -> pd.DataFrame:
    available = [
        strategy for strategy in selected_strategies
        if strategy in strategy_returns.columns
    ]

    common_start = max(
        first_valid_date(strategy_returns[strategy])
        for strategy in available
    )

    common_returns = strategy_returns.loc[common_start:, available].dropna(how="any")

    rows = []

    for strategy in available:
        metrics = compute_performance_metrics(common_returns[strategy])
        metrics["strategy"] = strategy
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows).set_index("strategy")
    metrics_df = metrics_df.sort_values("calmar", ascending=False)

    return metrics_df


def plot_equity_curves(returns: pd.DataFrame, filename: str, title: str) -> None:
    equity_curves = (1.0 + returns.fillna(0.0)).cumprod()

    plt.figure(figsize=(14, 7))

    for column in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[column], linewidth=1.5, label=column)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()


def plot_drawdowns(returns: pd.DataFrame, filename: str, title: str) -> None:
    drawdowns = compute_drawdown_series(returns)

    plt.figure(figsize=(14, 7))

    for column in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[column], linewidth=1.5, label=column)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    print("Loading data...")
    investable_returns = load_returns(returns_path)
    strategy_returns = load_strategy_returns()

    selected_existing_strategies = [
        "benchmark_static_6040_tc5bps",
        "benchmark_equal_weight_tc5bps",
        "hmm_k3_hrp_tc5bps",
        "hmm_k3_inverse_vol_tc5bps",
        "hmm_k3_risk_parity_tc5bps",
        "hmm_k4_hrp_tc5bps",
    ]

    print("\nComputing common-period metrics...")
    common_metrics = compute_common_period_metrics(
        strategy_returns=strategy_returns,
        selected_strategies=selected_existing_strategies,
    )

    common_metrics.to_csv(OUTPUT_DIR / "common_period_metrics.csv")

    print("\nCommon-period metrics:")
    display_cols = [
        "final_value",
        "cagr",
        "annual_volatility",
        "sharpe_0rf",
        "sortino_0rf",
        "max_drawdown",
        "calmar",
    ]
    print(common_metrics[display_cols])

    print("\nBuilding 60/40 + HMM overlays...")

    benchmark_weights = load_strategy_weights("benchmark_static_6040_tc0bps")

    hmm_source_strategies = [
        "hmm_k3_hrp_tc0bps",
        "hmm_k3_inverse_vol_tc0bps",
        "hmm_k3_risk_parity_tc0bps",
    ]

    hmm_alphas = [0.25, 0.50, 0.75]
    transaction_cost_bps = 5.0

    overlay_results = []

    for hmm_strategy in hmm_source_strategies:
        hmm_weights = load_strategy_weights(hmm_strategy)

        common_index = (
            investable_returns.index
            .intersection(benchmark_weights.index)
            .intersection(hmm_weights.index)
        )

        benchmark_aligned = benchmark_weights.reindex(common_index).ffill().fillna(0.0)
        hmm_aligned = hmm_weights.reindex(common_index).ffill().fillna(0.0)

        active_mask = (
            (benchmark_aligned.abs().sum(axis=1) > 0)
            & (hmm_aligned.abs().sum(axis=1) > 0)
        )

        benchmark_aligned = benchmark_aligned.loc[active_mask]
        hmm_aligned = hmm_aligned.loc[active_mask]

        for alpha in hmm_alphas:
            overlay_weights = (
                (1.0 - alpha) * benchmark_aligned
                + alpha * hmm_aligned
            )

            clean_hmm_name = hmm_strategy.replace("_tc0bps", "")
            strategy_name = (
                f"overlay_{int(alpha * 100)}pct_{clean_hmm_name}_tc5bps"
            )

            result = backtest_weight_schedule(
                strategy_name=strategy_name,
                returns=investable_returns,
                weights=overlay_weights,
                transaction_cost_bps=transaction_cost_bps,
            )

            result.returns.to_csv(OUTPUT_DIR / f"{strategy_name}_returns.csv")
            result.weights.to_csv(OUTPUT_DIR / f"{strategy_name}_weights.csv")

            overlay_results.append(result)

    overlay_returns = pd.concat(
        [result.returns for result in overlay_results],
        axis=1,
    )

    overlay_metrics = pd.DataFrame(
        [result.metrics for result in overlay_results]
    ).set_index("strategy")

    overlay_metrics = overlay_metrics.sort_values("calmar", ascending=False)

    overlay_returns.to_csv(OUTPUT_DIR / "overlay_strategy_returns.csv")
    overlay_metrics.to_csv(OUTPUT_DIR / "overlay_strategy_metrics.csv")

    print("\nOverlay metrics:")
    overlay_display_cols = [
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
    print(overlay_metrics[overlay_display_cols])

    print("\nBuilding comparison metrics and figures...")

    comparison_returns = pd.concat(
        [
            strategy_returns[
                [
                    "benchmark_static_6040_tc5bps",
                    "benchmark_equal_weight_tc5bps",
                    "hmm_k3_hrp_tc5bps",
                    "hmm_k3_inverse_vol_tc5bps",
                    "hmm_k3_risk_parity_tc5bps",
                ]
            ],
            overlay_returns,
        ],
        axis=1,
    )

    common_start = max(
        first_valid_date(comparison_returns[col])
        for col in comparison_returns.columns
    )

    comparison_returns = comparison_returns.loc[common_start:].dropna(how="any")

    comparison_metrics_rows = []

    for col in comparison_returns.columns:
        metrics = compute_performance_metrics(comparison_returns[col])
        metrics["strategy"] = col
        comparison_metrics_rows.append(metrics)

    comparison_metrics = (
        pd.DataFrame(comparison_metrics_rows)
        .set_index("strategy")
        .sort_values("calmar", ascending=False)
    )

    comparison_returns.to_csv(OUTPUT_DIR / "overlay_comparison_returns.csv")
    comparison_metrics.to_csv(OUTPUT_DIR / "overlay_comparison_metrics.csv")

    plot_equity_curves(
        returns=comparison_returns,
        filename="overlay_comparison_equity_curves.png",
        title="60/40 + HMM overlay strategies",
    )

    plot_drawdowns(
        returns=comparison_returns,
        filename="overlay_comparison_drawdowns.png",
        title="60/40 + HMM overlay drawdowns",
    )

    print("\nOverlay comparison metrics:")
    print(comparison_metrics[display_cols])

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()