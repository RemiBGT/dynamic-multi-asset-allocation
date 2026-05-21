from pathlib import Path
import sys

# Allows imports from src when running the script from the project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.data_loader import (  # noqa: E402
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    build_market_dataset,
    download_fred_series,
    download_yahoo_prices,
    load_config,
    save_dataframe,
)


def main() -> None:
    config = load_config()

    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    equity_tickers = config["data"]["investable_tickers"]["equity"]
    bond_tickers = config["data"]["investable_tickers"]["fixed_income"]
    investable_tickers = equity_tickers + bond_tickers

    yahoo_indicators = config["data"]["market_indicators"]["yahoo"]
    fred_series = config["data"]["fred_series"]

    print("Downloading investable ETF prices...")
    etf_prices = download_yahoo_prices(
        tickers=investable_tickers,
        start=start_date,
        end=end_date,
    )

    print("Downloading VIX...")
    vix_prices = download_yahoo_prices(
        tickers=yahoo_indicators,
        start=start_date,
        end=end_date,
    )

    print("Downloading FRED macro-financial data...")
    macro_data = download_fred_series(
        series_mapping=fred_series,
        start=start_date,
        end=end_date,
    )

    print("Building aligned market dataset...")
    market_dataset = build_market_dataset(
        etf_prices=etf_prices,
        vix_prices=vix_prices,
        macro_data=macro_data,
    )

    save_dataframe(etf_prices, RAW_DATA_DIR / "etf_prices.csv")
    save_dataframe(vix_prices, RAW_DATA_DIR / "vix.csv")
    save_dataframe(macro_data, RAW_DATA_DIR / "fred_macro.csv")
    save_dataframe(market_dataset, PROCESSED_DATA_DIR / "market_dataset.csv")

    print("\nDownload complete.")
    print(f"ETF prices shape: {etf_prices.shape}")
    print(f"VIX shape: {vix_prices.shape}")
    print(f"Macro data shape: {macro_data.shape}")
    print(f"Market dataset shape: {market_dataset.shape}")

    print("\nFiles saved:")
    print(f"- {RAW_DATA_DIR / 'etf_prices.csv'}")
    print(f"- {RAW_DATA_DIR / 'vix.csv'}")
    print(f"- {RAW_DATA_DIR / 'fred_macro.csv'}")
    print(f"- {PROCESSED_DATA_DIR / 'market_dataset.csv'}")


if __name__ == "__main__":
    main()