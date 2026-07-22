import yaml
import pandas as pd
import yfinance as yf
from pathlib import Path


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_ticker_data(data, ticker):
    """
    Handles yfinance output for multiple tickers.
    """
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            return data[ticker].copy()
        if ticker in data.columns.get_level_values(1):
            return data.xs(ticker, axis=1, level=1).copy()
    return data.copy()


def main():
    config = load_config()

    universe_file = Path(config["data"]["universe_file"])
    raw_prices_file = Path(config["data"]["raw_prices_file"])
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    benchmark = config["project"]["benchmark"]

    universe = pd.read_csv(universe_file)
    tickers = universe["ticker"].dropna().astype(str).str.upper().unique().tolist()
    all_tickers = sorted(list(set(tickers + [benchmark])))

    print(f"Downloading daily adjusted prices from {start_date} to {end_date}")
    print(f"Number of tickers including benchmark: {len(all_tickers)}")
    print(all_tickers)

    data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True
    )

    records = []

    for ticker in all_tickers:
        try:
            ticker_data = extract_ticker_data(data, ticker)
            ticker_data = ticker_data.reset_index()
            ticker_data["ticker"] = ticker
            ticker_data = ticker_data.rename(columns={"Date": "date"})

            required_cols = ["date", "ticker", "Open", "High", "Low", "Close", "Volume"]
            ticker_data = ticker_data[required_cols]
            ticker_data = ticker_data.dropna(subset=["Close"])

            records.append(ticker_data)
            print(f"{ticker}: {len(ticker_data)} rows")

        except Exception as e:
            print(f"Failed for {ticker}: {e}")

    if not records:
        raise RuntimeError("No data downloaded. Check ticker list, internet connection, or yfinance.")

    prices = pd.concat(records, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])

    raw_prices_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(raw_prices_file, index=False)

    print()
    print(f"Saved raw prices to: {raw_prices_file}")
    print(f"Rows: {len(prices)}")
    print(f"Tickers: {prices['ticker'].nunique()}")
    print(f"First date: {prices['date'].min().date()}")
    print(f"Last date: {prices['date'].max().date()}")


if __name__ == "__main__":
    main()
