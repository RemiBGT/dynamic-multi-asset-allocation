from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.features import (  # noqa: E402
    align_features_and_returns,
    build_investable_returns,
    build_regime_features,
    load_market_dataset,
)


def main() -> None:
    market_data_path = ROOT_DIR / "data" / "processed" / "market_dataset.csv"

    print("Loading market dataset...")
    market_data = load_market_dataset(market_data_path)

    print("Building regime features...")
    features = build_regime_features(market_data)

    print("Building investable returns...")
    returns = build_investable_returns(market_data)

    print("Aligning features and returns...")
    features, returns = align_features_and_returns(features, returns)

    features_path = ROOT_DIR / "data" / "processed" / "regime_features.csv"
    returns_path = ROOT_DIR / "data" / "processed" / "investable_returns.csv"

    features.to_csv(features_path)
    returns.to_csv(returns_path)

    print("\nFeature engineering complete.")
    print(f"Features shape: {features.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Start date: {features.index.min().date()}")
    print(f"End date: {features.index.max().date()}")

    print("\nFiles saved:")
    print(f"- {features_path}")
    print(f"- {returns_path}")

    print("\nMissing values in features:")
    print(features.isna().sum())

    print("\nMissing values in returns:")
    print(returns.isna().sum())


if __name__ == "__main__":
    main()