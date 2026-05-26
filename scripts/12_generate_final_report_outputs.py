from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.backtest import compute_drawdown_series  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "reports" / "final"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


FINAL_STRATEGIES = [
    "benchmark_static_6040_tc5bps",
    "benchmark_equal_weight_tc5bps",
    "hmm_k3_reduced_full_hrp_tc5bps",
    "hmm_k3_reduced_full_inverse_vol_tc5bps",
    "hmm_k3_reduced_full_risk_parity_tc5bps",
]


FINAL_STRATEGY_LABELS = {
    "benchmark_static_6040_tc5bps": "Static 60/40",
    "benchmark_equal_weight_tc5bps": "Static Equal Weight",
    "hmm_k3_reduced_full_hrp_tc5bps": "HMM K=3 + HRP",
    "hmm_k3_reduced_full_inverse_vol_tc5bps": "HMM K=3 + Inverse Vol",
    "hmm_k3_reduced_full_risk_parity_tc5bps": "HMM K=3 + Risk Parity",
}


FINAL_STRATEGY = "hmm_k3_reduced_full_hrp_tc5bps"
BENCHMARK_STRATEGY = "benchmark_static_6040_tc5bps"


METRIC_COLUMNS = [
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

def load_final_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load returns and metrics from the reduced-features HMM backtest.
    """
    returns_path = (
        ROOT_DIR
        / "reports"
        / "backtests"
        / "reduced_features"
        / "reduced_features_returns_common_period.csv"
    )

    metrics_path = (
        ROOT_DIR
        / "reports"
        / "backtests"
        / "reduced_features"
        / "reduced_features_metrics_common_period.csv"
    )

    strategy_returns = pd.read_csv(
        returns_path,
        index_col="date",
        parse_dates=True,
    ).sort_index()

    strategy_metrics = pd.read_csv(
        metrics_path,
        index_col=0,
    )

    return strategy_returns, strategy_metrics


def format_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean display table for the README.
    """
    selected = metrics.loc[FINAL_STRATEGIES, METRIC_COLUMNS].copy()
    selected.index = [FINAL_STRATEGY_LABELS[idx] for idx in selected.index]

    formatted = pd.DataFrame(index=selected.index)

    formatted["Final value"] = selected["final_value"].map(lambda x: f"{x:.2f}")
    formatted["CAGR"] = selected["cagr"].map(lambda x: f"{x:.2%}")
    formatted["Volatility"] = selected["annual_volatility"].map(lambda x: f"{x:.2%}")
    formatted["Sharpe"] = selected["sharpe_0rf"].map(lambda x: f"{x:.2f}")
    formatted["Sortino"] = selected["sortino_0rf"].map(lambda x: f"{x:.2f}")
    formatted["Max drawdown"] = selected["max_drawdown"].map(lambda x: f"{x:.2%}")
    formatted["Calmar"] = selected["calmar"].map(lambda x: f"{x:.2f}")
    formatted["Turnover"] = selected["total_turnover"].map(lambda x: f"{x:.2f}")
    formatted["Transaction costs"] = selected["total_transaction_costs"].map(
        lambda x: f"{x:.2%}"
    )

    return formatted


def save_metric_tables(metrics: pd.DataFrame) -> None:
    """
    Save final tables as CSV and Markdown.
    """
    raw_table = metrics.loc[FINAL_STRATEGIES, METRIC_COLUMNS].copy()
    raw_table.index = [FINAL_STRATEGY_LABELS[idx] for idx in raw_table.index]

    formatted_table = format_metric_table(metrics)

    raw_table.to_csv(TABLE_DIR / "final_metrics_raw.csv")
    formatted_table.to_csv(TABLE_DIR / "final_metrics_formatted.csv")

    markdown_table = formatted_table.to_markdown()

    with open(TABLE_DIR / "final_metrics_table.md", "w", encoding="utf-8") as file:
        file.write(markdown_table)

    print("\nFinal formatted metrics:")
    print(formatted_table)

def get_final_returns(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only final strategies and rename columns for display.
    """
    selected = strategy_returns[FINAL_STRATEGIES].copy()
    selected = selected.rename(columns=FINAL_STRATEGY_LABELS)
    return selected


def plot_equity_curves(strategy_returns: pd.DataFrame) -> None:
    """
    Plot cumulative returns of the final strategies.
    """
    final_returns = get_final_returns(strategy_returns)
    equity_curves = (1.0 + final_returns.fillna(0.0)).cumprod()

    plt.figure(figsize=(14, 7))

    for column in equity_curves.columns:
        linewidth = 2.4 if column == FINAL_STRATEGY_LABELS[FINAL_STRATEGY] else 1.4
        plt.plot(
            equity_curves.index,
            equity_curves[column],
            linewidth=linewidth,
            label=column,
        )

    plt.title("Final strategy vs benchmarks — cumulative performance")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "final_equity_curves.png", dpi=180)
    plt.close()


def plot_drawdowns(strategy_returns: pd.DataFrame) -> None:
    """
    Plot drawdowns of the final strategies.
    """
    final_returns = get_final_returns(strategy_returns)
    drawdowns = compute_drawdown_series(final_returns)

    plt.figure(figsize=(14, 7))

    for column in drawdowns.columns:
        linewidth = 2.4 if column == FINAL_STRATEGY_LABELS[FINAL_STRATEGY] else 1.4
        plt.plot(
            drawdowns.index,
            drawdowns[column],
            linewidth=linewidth,
            label=column,
        )

    plt.title("Final strategy vs benchmarks — drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "final_drawdowns.png", dpi=180)
    plt.close()


def plot_metric_barplots(metrics: pd.DataFrame) -> None:
    """
    Plot key metrics as bar charts.
    """
    selected = metrics.loc[FINAL_STRATEGIES].copy()
    selected.index = [FINAL_STRATEGY_LABELS[idx] for idx in selected.index]

    metrics_to_plot = {
        "cagr": "CAGR",
        "annual_volatility": "Annualized volatility",
        "sharpe_0rf": "Sharpe ratio",
        "max_drawdown": "Maximum drawdown",
        "calmar": "Calmar ratio",
    }

    for metric, title in metrics_to_plot.items():
        plt.figure(figsize=(12, 6))
        values = selected[metric]

        plt.bar(values.index, values.values)
        plt.title(title)
        plt.xlabel("Strategy")
        plt.ylabel(title)
        plt.xticks(rotation=30, ha="right")

        if metric in {"cagr", "annual_volatility", "max_drawdown"}:
            plt.gca().yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.0%}")
            )

        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"final_metric_{metric}.png", dpi=180)
        plt.close()

def load_final_weights() -> pd.DataFrame:
    """
    Load the daily weights of the final HMM + HRP strategy.
    """
    weights_path = (
        ROOT_DIR
        / "reports"
        / "backtests"
        / "reduced_features"
        / f"{FINAL_STRATEGY}_weights.csv"
    )

    weights = pd.read_csv(
        weights_path,
        index_col="date",
        parse_dates=True,
    ).sort_index()

    return weights


def plot_final_weights(weights: pd.DataFrame) -> None:
    """
    Plot the dynamic asset weights of the final strategy.
    """
    weights = weights.loc[:, weights.abs().sum(axis=0) > 0].copy()

    plt.figure(figsize=(14, 7))
    plt.stackplot(
        weights.index,
        [weights[col] for col in weights.columns],
        labels=weights.columns,
    )

    plt.title("Final HMM + HRP strategy — dynamic asset allocation")
    plt.xlabel("Date")
    plt.ylabel("Portfolio weight")
    plt.legend(loc="upper left", fontsize=8, ncol=3)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "final_dynamic_weights.png", dpi=180)
    plt.close()


def load_final_regimes() -> pd.DataFrame:
    """
    Load the reduced-feature HMM regime path used by the final strategy.
    """
    regimes_path = (
        ROOT_DIR
        / "reports"
        / "hmm"
        / "reduced_features"
        / "walk_forward_reduced_full.csv"
    )

    regimes = pd.read_csv(
        regimes_path,
        index_col="date",
        parse_dates=True,
    ).sort_index()

    regimes = regimes[regimes["error"].fillna("") == ""].copy()

    return regimes


def plot_regime_timeline(regimes: pd.DataFrame) -> None:
    """
    Plot the walk-forward HMM regime labels through time.
    """
    labels = regimes["current_regime_label"].astype("category")
    label_codes = labels.cat.codes

    plt.figure(figsize=(14, 4))
    plt.step(regimes.index, label_codes, where="post")

    plt.yticks(
        range(len(labels.cat.categories)),
        labels.cat.categories,
    )

    plt.title("Reduced-feature HMM K=3 — walk-forward regime timeline")
    plt.xlabel("Date")
    plt.ylabel("Detected regime")
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "final_regime_timeline.png", dpi=180)
    plt.close()

def copy_feature_correlation_heatmap() -> None:
    """
    Copy the feature correlation heatmap into the final report folder.
    """
    source_path = (
        ROOT_DIR
        / "reports"
        / "features"
        / "feature_correlation_heatmap.png"
    )

    target_path = FIGURE_DIR / "final_feature_correlation_heatmap.png"

    if source_path.exists():
        shutil.copy(source_path, target_path)
    else:
        print(
            "Warning: feature correlation heatmap not found. "
            "Run scripts/10_feature_correlation_analysis.py first."
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading final inputs...")
    strategy_returns, strategy_metrics = load_final_inputs()

    print("Saving final metric tables...")
    save_metric_tables(strategy_metrics)

    print("Generating final equity curves...")
    plot_equity_curves(strategy_returns)

    print("Generating final drawdowns...")
    plot_drawdowns(strategy_returns)

    print("Generating final metric barplots...")
    plot_metric_barplots(strategy_metrics)

    print("Loading and plotting final dynamic weights...")
    final_weights = load_final_weights()
    plot_final_weights(final_weights)

    print("Loading and plotting final regime timeline...")
    final_regimes = load_final_regimes()
    plot_regime_timeline(final_regimes)

    print("Copying feature correlation heatmap...")
    copy_feature_correlation_heatmap()

    print("\nFinal report outputs generated.")
    print(f"Figures saved in: {FIGURE_DIR}")
    print(f"Tables saved in: {TABLE_DIR}")


if __name__ == "__main__":
    main()