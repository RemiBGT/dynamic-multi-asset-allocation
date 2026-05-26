from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.hmm_regimes import CORE_HMM_FEATURES, load_features  # noqa: E402


OUTPUT_DIR = ROOT_DIR / "reports" / "features"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features_path = ROOT_DIR / "data" / "processed" / "regime_features.csv"

    features = load_features(features_path)
    features = features[CORE_HMM_FEATURES].dropna()

    corr = features.corr()

    corr.to_csv(OUTPUT_DIR / "feature_correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=8)

    plt.title("HMM feature correlation matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation_heatmap.png", dpi=150)
    plt.close()

    pairs = []

    for i, col_i in enumerate(corr.columns):
        for j, col_j in enumerate(corr.columns):
            if j <= i:
                continue

            pairs.append(
                {
                    "feature_1": col_i,
                    "feature_2": col_j,
                    "correlation": corr.loc[col_i, col_j],
                    "abs_correlation": abs(corr.loc[col_i, col_j]),
                }
            )

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values("abs_correlation", ascending=False)

    pairs_df.to_csv(OUTPUT_DIR / "feature_correlation_pairs.csv", index=False)

    print("Correlation analysis complete.")
    print(f"Outputs saved in: {OUTPUT_DIR}")

    print("\nTop correlated feature pairs:")
    print(pairs_df.head(20))


if __name__ == "__main__":
    main()