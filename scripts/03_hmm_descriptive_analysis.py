from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.hmm_regimes import (  # noqa: E402
    CORE_HMM_FEATURES,
    add_economic_labels,
    build_model_selection_row,
    compute_regime_durations,
    compute_regime_feature_summary,
    compute_regime_return_summary,
    compute_transition_matrix_from_states,
    fit_gaussian_hmm,
    load_features,
    load_returns,
    select_features,
    suggest_regime_labels,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "hmm" / "descriptive"
FIGURE_DIR = OUTPUT_DIR / "figures"


def plot_regimes_on_spy(
    states: pd.Series,
    labels: pd.Series,
    returns: pd.DataFrame,
    n_components: int,
) -> None:
    """
    Plot SPY cumulative performance and color points by regime.
    """
    if "SPY" not in returns.columns:
        return

    common_index = states.index.intersection(returns.index)

    spy_curve = (1.0 + returns.loc[common_index, "SPY"]).cumprod()
    states = states.loc[common_index]
    labels = labels.loc[common_index]

    plt.figure(figsize=(14, 7))
    plt.plot(
        spy_curve.index,
        spy_curve.values,
        linewidth=1.5,
        label="SPY cumulative return",
    )

    for regime in sorted(states.unique()):
        mask = states == regime
        label = labels[mask].iloc[0]

        plt.scatter(
            spy_curve.index[mask],
            spy_curve.loc[mask],
            s=8,
            label=f"Regime {regime}: {label}",
        )

    plt.title(f"SPY cumulative return colored by HMM regime - K={n_components}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    path = FIGURE_DIR / f"spy_regimes_k{n_components}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_transition_matrix(
    transition_matrix: pd.DataFrame,
    n_components: int,
) -> None:
    """
    Plot empirical transition matrix computed from decoded states.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(transition_matrix.values, aspect="auto")
    plt.colorbar(label="Transition probability")

    plt.xticks(
        range(len(transition_matrix.columns)),
        transition_matrix.columns,
        rotation=45,
    )
    plt.yticks(
        range(len(transition_matrix.index)),
        transition_matrix.index,
    )

    plt.title(f"Empirical transition matrix - K={n_components}")
    plt.xlabel("To regime")
    plt.ylabel("From regime")

    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{transition_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    path = FIGURE_DIR / f"transition_matrix_k{n_components}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    features_path = ROOT_DIR / "data" / "processed" / "regime_features.csv"
    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    print("Loading features and returns...")
    raw_features = load_features(features_path)
    returns = load_returns(returns_path)

    features = select_features(raw_features, CORE_HMM_FEATURES)

    print(f"Selected features shape: {features.shape}")
    print("Selected features:")
    for feature in features.columns:
        print(f"  - {feature}")

    model_selection_rows = []

    for n_components in range(2, 6):
        print(f"\nFitting descriptive HMM with K={n_components}...")

        fit_result = fit_gaussian_hmm(
            features=features,
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
        )

        model_selection_rows.append(
            build_model_selection_row(
                fit_result=fit_result,
                n_components=n_components,
                n_features=features.shape[1],
                covariance_type="full",
            )
        )

        feature_summary = compute_regime_feature_summary(
            features,
            fit_result.states,
        )

        return_summary = compute_regime_return_summary(
            returns,
            fit_result.states,
        )

        transition_matrix = compute_transition_matrix_from_states(
            fit_result.states,
        )

        durations = compute_regime_durations(fit_result.states)

        labels = suggest_regime_labels(feature_summary)
        regime_labels = add_economic_labels(fit_result.states, labels)

        regime_path = pd.concat(
            [
                fit_result.states,
                regime_labels,
                fit_result.probabilities,
            ],
            axis=1,
        )

        feature_summary["suggested_label"] = feature_summary.index.map(labels)
        return_summary["suggested_label"] = return_summary.index.map(labels)
        durations["suggested_label"] = durations.index.map(labels)

        feature_summary.to_csv(
            OUTPUT_DIR / f"regime_feature_summary_k{n_components}.csv"
        )
        return_summary.to_csv(
            OUTPUT_DIR / f"regime_return_summary_k{n_components}.csv"
        )
        transition_matrix.to_csv(
            OUTPUT_DIR / f"transition_matrix_k{n_components}.csv"
        )
        durations.to_csv(
            OUTPUT_DIR / f"regime_durations_k{n_components}.csv"
        )
        regime_path.to_csv(
            OUTPUT_DIR / f"regime_path_k{n_components}.csv"
        )

        plot_regimes_on_spy(
            states=fit_result.states,
            labels=regime_labels,
            returns=returns,
            n_components=n_components,
        )

        plot_transition_matrix(
            transition_matrix=transition_matrix,
            n_components=n_components,
        )

        print("Model converged:", fit_result.converged)
        print("Log-likelihood:", round(fit_result.log_likelihood, 2))

        print("\nSuggested labels:")
        for regime, label in labels.items():
            print(f"  Regime {regime}: {label}")

        print("\nRegime frequencies:")
        print(feature_summary[["n_obs", "frequency", "suggested_label"]])

    model_selection = pd.DataFrame(model_selection_rows)
    model_selection.to_csv(OUTPUT_DIR / "model_selection.csv", index=False)

    print("\nDescriptive HMM analysis complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    print("\nModel selection:")
    print(model_selection)


if __name__ == "__main__":
    main()