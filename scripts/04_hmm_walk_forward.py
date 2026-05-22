from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.hmm_regimes import (  # noqa: E402
    CORE_HMM_FEATURES,
    load_features,
    select_features,
    summarize_walk_forward_results,
    walk_forward_hmm_detection,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "hmm" / "walk_forward"
FIGURE_DIR = OUTPUT_DIR / "figures"


def plot_walk_forward_labels(
    wf_results: pd.DataFrame,
    n_components: int,
) -> None:
    valid = wf_results[wf_results["error"].fillna("") == ""].copy()

    if valid.empty:
        return

    labels = valid["current_regime_label"].astype("category")
    label_codes = labels.cat.codes

    plt.figure(figsize=(14, 4))
    plt.step(valid.index, label_codes, where="post")

    plt.yticks(
        range(len(labels.cat.categories)),
        labels.cat.categories,
    )

    plt.title(f"Walk-forward HMM regime labels - K={n_components}")
    plt.xlabel("Date")
    plt.ylabel("Regime label")
    plt.tight_layout()

    path = FIGURE_DIR / f"walk_forward_labels_k{n_components}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def plot_walk_forward_confidence(
    wf_results: pd.DataFrame,
    n_components: int,
) -> None:
    valid = wf_results[wf_results["error"].fillna("") == ""].copy()

    if valid.empty:
        return

    plt.figure(figsize=(14, 4))
    plt.plot(
        valid.index,
        valid["current_regime_probability"],
        linewidth=1.5,
    )

    plt.axhline(0.50, linestyle="--", linewidth=1)
    plt.axhline(0.65, linestyle="--", linewidth=1)

    plt.title(f"Walk-forward current regime probability - K={n_components}")
    plt.xlabel("Date")
    plt.ylabel("Probability of dominant regime")
    plt.tight_layout()

    path = FIGURE_DIR / f"walk_forward_confidence_k{n_components}.png"
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    features_path = ROOT_DIR / "data" / "processed" / "regime_features.csv"

    print("Loading regime features...")
    raw_features = load_features(features_path)
    features = select_features(raw_features, CORE_HMM_FEATURES)

    print(f"Selected features shape: {features.shape}")

    summary_rows = []

    for n_components in range(2, 6):
        print(f"\nRunning walk-forward HMM detection with K={n_components}...")

        wf_results = walk_forward_hmm_detection(
            features=features,
            n_components=n_components,
            train_years=5,
            covariance_type="full",
            min_train_observations=756,
            n_iter=1000,
        )

        output_path = OUTPUT_DIR / f"walk_forward_results_k{n_components}.csv"
        wf_results.to_csv(output_path)

        summary = summarize_walk_forward_results(wf_results)
        summary["n_components"] = n_components
        summary_rows.append(summary)

        plot_walk_forward_labels(wf_results, n_components)
        plot_walk_forward_confidence(wf_results, n_components)

        print(f"Saved: {output_path}")
        print(pd.Series(summary))

        if not wf_results.empty:
            valid = wf_results[wf_results["error"].fillna("") == ""]
            if not valid.empty:
                print("\nRegime label counts:")
                print(valid["current_regime_label"].value_counts())

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df[
        [
            "n_components",
            "n_decision_dates",
            "mean_current_regime_probability",
            "median_current_regime_probability",
            "min_current_regime_probability",
            "n_regime_switches",
            "switch_rate",
        ]
    ]

    summary_df.to_csv(OUTPUT_DIR / "walk_forward_summary.csv", index=False)

    print("\nWalk-forward HMM analysis complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()