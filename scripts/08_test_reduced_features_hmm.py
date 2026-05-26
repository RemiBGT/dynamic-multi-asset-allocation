from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.hmm_regimes import (  # noqa: E402
    compute_regime_feature_summary,
    load_features,
    summarize_walk_forward_results,
    walk_forward_hmm_detection,
)


OUTPUT_DIR = ROOT_DIR / "reports" / "hmm" / "reduced_features"


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
    """
    Select a parsimonious feature set for HMM regime detection.
    """
    missing_features = [
        feature for feature in REDUCED_HMM_FEATURES
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(f"Missing reduced HMM features: {missing_features}")

    selected = features[REDUCED_HMM_FEATURES].copy()
    selected = selected.replace([float("inf"), float("-inf")], pd.NA).dropna()

    return selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = ROOT_DIR / "data" / "processed" / "regime_features.csv"

    print("Loading regime features...")
    raw_features = load_features(features_path)
    features = select_reduced_features(raw_features)

    print(f"Reduced features shape: {features.shape}")
    print("Reduced feature set:")
    for feature in features.columns:
        print(f"  - {feature}")

    summary_rows = []

    experiments = [
        {
            "experiment": "reduced_full",
            "n_components": 3,
            "covariance_type": "full",
        },
        {
            "experiment": "reduced_diag",
            "n_components": 3,
            "covariance_type": "diag",
        },
    ]

    for experiment_config in experiments:
        experiment_name = experiment_config["experiment"]
        n_components = experiment_config["n_components"]
        covariance_type = experiment_config["covariance_type"]

        print(
            f"\nRunning experiment={experiment_name}, "
            f"K={n_components}, covariance_type={covariance_type}..."
        )

        wf_results = walk_forward_hmm_detection(
            features=features,
            n_components=n_components,
            train_years=5,
            covariance_type=covariance_type,
            min_train_observations=756,
            n_iter=1000,
        )

        output_path = OUTPUT_DIR / f"walk_forward_{experiment_name}.csv"
        wf_results.to_csv(output_path)

        summary = summarize_walk_forward_results(wf_results)
        summary["experiment"] = experiment_name
        summary["n_components"] = n_components
        summary["covariance_type"] = covariance_type

        summary_rows.append(summary)

        valid = wf_results[wf_results["error"].fillna("") == ""].copy()

        print(f"Saved: {output_path}")
        print(pd.Series(summary))

        if not valid.empty:
            print("\nRegime label counts:")
            print(valid["current_regime_label"].value_counts())

            probability_cols = [
                col for col in valid.columns
                if col.startswith("current_prob_regime_")
            ]

            if probability_cols:
                max_probability = valid[probability_cols].max(axis=1)

                print("\nDominant regime probability diagnostics:")
                print(max_probability.describe())

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            "experiment",
            "n_components",
            "covariance_type",
            "n_decision_dates",
            "mean_current_regime_probability",
            "median_current_regime_probability",
            "min_current_regime_probability",
            "n_regime_switches",
            "switch_rate",
        ]
    ]

    summary_df.to_csv(OUTPUT_DIR / "reduced_features_summary.csv", index=False)

    print("\nReduced-features HMM experiments complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()