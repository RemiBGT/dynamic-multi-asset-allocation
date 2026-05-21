from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
import yaml
from pandas_datareader import data as pdr


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load project configuration from YAML.
    """
    if config_path is None:
        config_path = ROOT_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_yahoo_prices(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance using yfinance.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError("No data downloaded from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    prices = prices.sort_index()
    prices.index.name = "date"

    return prices.dropna(how="all")


def download_fred_series(
    series_mapping: Dict[str, str],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download macro-financial series from FRED.

    Parameters
    ----------
    series_mapping:
        Dictionary where keys are FRED tickers and values are clean column names.
    """
    fred_codes = list(series_mapping.keys())

    macro = pdr.DataReader(
        fred_codes,
        data_source="fred",
        start=start,
        end=end,
    )

    macro = macro.rename(columns=series_mapping)
    macro = macro.sort_index()
    macro.index.name = "date"

    return macro.dropna(how="all")


def build_market_dataset(
    etf_prices: pd.DataFrame,
    vix_prices: pd.DataFrame,
    macro_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align ETF prices, VIX and macro data on the ETF trading calendar.
    """
    vix = vix_prices.copy()

    if "^VIX" in vix.columns:
        vix = vix.rename(columns={"^VIX": "vix"})
    elif len(vix.columns) == 1:
        vix.columns = ["vix"]

    macro_aligned = macro_data.reindex(etf_prices.index).ffill()
    vix_aligned = vix.reindex(etf_prices.index).ffill()

    market_dataset = pd.concat(
        [
            etf_prices.add_prefix("px_"),
            vix_aligned,
            macro_aligned,
        ],
        axis=1,
    )

    market_dataset.index.name = "date"

    return market_dataset.dropna(how="all")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save a dataframe to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)