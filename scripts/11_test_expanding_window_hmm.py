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
)
from src.hmm_regimes import (  # noqa: E402
    load_features,
    summarize_walk_forward_results,
    walk_forward_hmm_detection,
    walk_forward_hmm_detection_expanding,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "hmm" / "training_windows"


REDUCED_HMM_FEATURES = [
    "spy_ret_63d",
    "spy_vol_21d",
    "tlt_ret_21d",
    "hyg_ret_21d",
    "spy_tlt_corr_63d",
    "vix_level",
    "us_10y_change_21d",
    "real_yield_10y_change_21d",
    "credit_quality_spread",
]


def select_reduced_features(features: pd.DataFrame) -> pd.DataFrame:
    missing_features = [
        feature for feature in REDUCED_HMM_FEATURES
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(f"Missing reduced HMM features: {missing_features}")

    selected = features[REDUCED_HMM_FEATURES].copy()
    selected = selected.replace([float("inf"), float("-inf")], pd.NA).dropna()

    return selected


def rename_result(
    result: BacktestResult,
    new_name: str,
    hmm_experiment: str,
) -> BacktestResult:
    result.strategy_name = new_name
    result.returns = result.returns.rename(new_name)
    result.gross_returns = result.gross_returns.rename(new_name + "_gross")
    result.metrics["strategy"] = new_name
    result.metrics["hmm_experiment"] = hmm_experiment
    return result


def compute_common_period_metrics(
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
        metrics["hmm_experiment"] = result.metrics.get("hmm_experiment")
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


def backtest_hmm_experiment(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    experiment_name: str,
    n_components: int = 3,
    allocation_method: str = "hrp",
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """
    Backtest one HMM regime experiment using the same allocation method.
    """
    result = backtest_regime_strategy(
        returns=returns,
        regimes=regimes,
        n_components=n_components,
        allocation_method=allocation_method,
        transaction_cost_bps=transaction_cost_bps,
        lookback_days=252,
    )

    new_name = (
        f"hmm_k{n_components}_{experiment_name}"
        f"_{allocation_method}_tc{transaction_cost_bps:.0f}bps"
    )

    result = rename_result(
        result=result,
        new_name=new_name,
        hmm_experiment=experiment_name,
    )

    return result

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = ROOT_DIR / "data" / "processed" / "regime_features.csv"
    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    print("Loading data...")
    raw_features = load_features(features_path)
    features = select_reduced_features(raw_features)
    returns = load_returns(returns_path)

    print(f"Reduced features shape: {features.shape}")

    hmm_experiments = [
        {
            "name": "reduced_rolling_5y",
            "mode": "rolling",
            "train_years": 5,
            "min_train_observations": 1260,
        },
        {
            "name": "reduced_rolling_7y",
            "mode": "rolling",
            "train_years": 7,
            "min_train_observations": 1764,
        },
        {
            "name": "reduced_rolling_10y",
            "mode": "rolling",
            "train_years": 10,
            "min_train_observations": 2400,
        },
        {
            "name": "reduced_expanding_min_5y",
            "mode": "expanding",
            "min_train_years": 5,
            "min_train_observations": 1260,
        },
    ]

    all_results = []
    hmm_summary_rows = []

    print("\nBacktesting benchmarks...")

    for benchmark_method in ["static_6040", "equal_weight"]:
        result = backtest_benchmark_strategy(
            returns=returns,
            benchmark_method=benchmark_method,
            transaction_cost_bps=5.0,
            lookback_days=252,
        )

        result.metrics["hmm_experiment"] = "benchmark"
        all_results.append(result)

    for experiment in hmm_experiments:
        experiment_name = experiment["name"]
        mode = experiment["mode"]

        print(f"\nRunning HMM experiment: {experiment_name}")

        if mode == "rolling":
            regimes = walk_forward_hmm_detection(
                features=features,
                n_components=3,
                train_years=experiment["train_years"],
                covariance_type="full",
                min_train_observations=experiment["min_train_observations"],
                n_iter=1000,
            )

        elif mode == "expanding":
            regimes = walk_forward_hmm_detection_expanding(
                features=features,
                n_components=3,
                min_train_years=experiment["min_train_years"],
                covariance_type="full",
                min_train_observations=experiment["min_train_observations"],
                n_iter=1000,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        regimes_path = OUTPUT_DIR / f"walk_forward_{experiment_name}.csv"
        regimes.to_csv(regimes_path)

        hmm_summary = summarize_walk_forward_results(regimes)
        hmm_summary["experiment"] = experiment_name
        hmm_summary["mode"] = mode
        hmm_summary_rows.append(hmm_summary)

        print("HMM summary:")
        print(pd.Series(hmm_summary))

        valid = regimes[regimes["error"].fillna("") == ""].copy()

        if not valid.empty:
            print("\nRegime label counts:")
            print(valid["current_regime_label"].value_counts())

        print(f"\nBacktesting allocation for {experiment_name}...")

        result = backtest_hmm_experiment(
            returns=returns,
            regimes=regimes,
            experiment_name=experiment_name,
            n_components=3,
            allocation_method="hrp",
            transaction_cost_bps=5.0,
        )

        result.returns.to_csv(OUTPUT_DIR / f"{result.strategy_name}_returns.csv")
        result.weights.to_csv(OUTPUT_DIR / f"{result.strategy_name}_weights.csv")

        all_results.append(result)

    hmm_summary_df = pd.DataFrame(hmm_summary_rows)
    hmm_summary_df.to_csv(OUTPUT_DIR / "training_window_hmm_summary.csv", index=False)

    print("\nComputing common-period metrics...")

    common_returns, common_metrics = compute_common_period_metrics(all_results)

    common_returns.to_csv(OUTPUT_DIR / "training_window_returns_common_period.csv")
    common_metrics.to_csv(OUTPUT_DIR / "training_window_metrics_common_period.csv")

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

    print("\nTraining-window HMM backtest complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    print("\nHMM regime-detection summary:")
    print(hmm_summary_df)

    print("\nTop strategies by Calmar:")
    print(common_metrics[available_cols].sort_values("calmar", ascending=False))

    print("\nTop strategies by Sharpe:")
    print(common_metrics[available_cols].sort_values("sharpe_0rf", ascending=False))


if __name__ == "__main__":
    main()