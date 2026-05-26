from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.backtest import (  # noqa: E402
    BacktestResult,
    backtest_benchmark_strategy,
    backtest_regime_strategy,
    combine_strategy_returns,
    compute_performance_metrics,
    load_returns,
    load_walk_forward_regimes,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "backtests" / "reduced_features"


def rename_result(result: BacktestResult, new_name: str, experiment: str) -> BacktestResult:
    result.strategy_name = new_name
    result.returns = result.returns.rename(new_name)
    result.gross_returns = result.gross_returns.rename(new_name + "_gross")
    result.metrics["strategy"] = new_name
    result.metrics["hmm_experiment"] = experiment
    return result


def compute_common_period_metrics(results: list[BacktestResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        metrics["hmm_experiment"] = result.metrics.get("hmm_experiment", "benchmark")
        metrics["n_components"] = result.metrics.get("n_components")
        metrics["allocation_method"] = result.metrics.get("allocation_method")
        metrics["transaction_cost_bps"] = result.metrics.get("transaction_cost_bps")

        rows.append(metrics)

    common_metrics = (
        pd.DataFrame(rows)
        .set_index("strategy")
        .sort_values("calmar", ascending=False)
    )

    return common_returns, common_metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    print("Loading investable returns...")
    returns = load_returns(returns_path)

    all_results = []

    print("Backtesting benchmarks...")

    for benchmark_method in ["static_6040", "equal_weight"]:
        result = backtest_benchmark_strategy(
            returns=returns,
            benchmark_method=benchmark_method,
            transaction_cost_bps=5.0,
            lookback_days=252,
        )

        result.metrics["hmm_experiment"] = "benchmark"
        all_results.append(result)

    experiments = {
        "reduced_full": ROOT_DIR / "reports" / "hmm" / "reduced_features" / "walk_forward_reduced_full.csv",
        "reduced_diag": ROOT_DIR / "reports" / "hmm" / "reduced_features" / "walk_forward_reduced_diag.csv",
    }

    allocation_methods = [
        "hrp",
        "inverse_vol",
        "risk_parity",
    ]

    for experiment_name, regimes_path in experiments.items():
        print(f"\nLoading regimes for experiment: {experiment_name}")
        regimes = load_walk_forward_regimes(regimes_path)

        for method in allocation_methods:
            print(f"Backtesting {experiment_name}, method={method}...")

            result = backtest_regime_strategy(
                returns=returns,
                regimes=regimes,
                n_components=3,
                allocation_method=method,
                transaction_cost_bps=5.0,
                lookback_days=252,
            )

            new_name = f"hmm_k3_{experiment_name}_{method}_tc5bps"

            result = rename_result(
                result=result,
                new_name=new_name,
                experiment=experiment_name,
            )

            result.returns.to_csv(OUTPUT_DIR / f"{new_name}_returns.csv")
            result.weights.to_csv(OUTPUT_DIR / f"{new_name}_weights.csv")

            all_results.append(result)

    print("\nComputing common-period metrics...")

    common_returns, common_metrics = compute_common_period_metrics(all_results)

    common_returns.to_csv(OUTPUT_DIR / "reduced_features_returns_common_period.csv")
    common_metrics.to_csv(OUTPUT_DIR / "reduced_features_metrics_common_period.csv")

    display_cols = [
        "hmm_experiment",
        "n_components",
        "allocation_method",
        "transaction_cost_bps",
        "final_value",
        "cagr",
        "annual_volatility",
        "sharpe_0rf",
        "max_drawdown",
        "calmar",
        "total_turnover",
        "total_transaction_costs",
    ]

    available_cols = [col for col in display_cols if col in common_metrics.columns]

    print("\nReduced-features HMM backtest complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    print("\nTop strategies by Calmar:")
    print(common_metrics[available_cols].sort_values("calmar", ascending=False))

    print("\nTop strategies by Sharpe:")
    print(common_metrics[available_cols].sort_values("sharpe_0rf", ascending=False))


if __name__ == "__main__":
    main()