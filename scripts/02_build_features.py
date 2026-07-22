import yaml
import pandas as pd
import numpy as np
from pathlib import Path


TARGET_COL = "outperform_qqq_next_1m"


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def max_drawdown(return_series):
    cumulative = (1 + return_series).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    return drawdown.min()


def standardise_price_columns(prices):
    prices = prices.copy()

    rename_map = {}
    for col in prices.columns:
        lower = col.lower()
        if lower == "date":
            rename_map[col] = "date"
        elif lower == "ticker":
            rename_map[col] = "ticker"
        elif lower == "open":
            rename_map[col] = "open"
        elif lower == "high":
            rename_map[col] = "high"
        elif lower == "low":
            rename_map[col] = "low"
        elif lower == "close":
            rename_map[col] = "close"
        elif lower == "volume":
            rename_map[col] = "volume"

    prices = prices.rename(columns=rename_map)

    required_cols = ["date", "ticker", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in prices.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in prices.csv: {missing_cols}")

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()

    return prices


def build_model_panel(prices, benchmark):
    prices = prices.sort_values(["ticker", "date"])

    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = prices.pivot(index="date", columns="ticker", values="volume").sort_index()

    if benchmark not in close.columns:
        raise ValueError(f"Benchmark {benchmark} was not found in price data.")

    daily_returns = close.pct_change(fill_method=None)

    monthly_close = close.resample("M").last()
    monthly_volume = volume.resample("M").last()
    monthly_returns = monthly_close.pct_change(fill_method=None)

    qqq_monthly_return = monthly_returns[benchmark]
    qqq_daily_return = daily_returns[benchmark]

    tickers = [ticker for ticker in monthly_close.columns if ticker != benchmark]

    feature_frames = []

    for ticker in tickers:
        df = pd.DataFrame(index=monthly_close.index)
        df["ticker"] = ticker

        # Momentum features
        df["ret_1m"] = monthly_close[ticker].pct_change(1, fill_method=None)
        df["ret_3m"] = monthly_close[ticker].pct_change(3, fill_method=None)
        df["ret_6m"] = monthly_close[ticker].pct_change(6, fill_method=None)
        df["ret_12m"] = monthly_close[ticker].pct_change(12, fill_method=None)

        # Volume feature
        df["volume_growth_3m"] = monthly_volume[ticker].pct_change(3, fill_method=None)

        # Daily return series for rolling risk features
        stock_daily_return = daily_returns[ticker]

        df["volatility_1m"] = (
            stock_daily_return.rolling(21).std() * np.sqrt(252)
        ).resample("M").last()

        df["volatility_3m"] = (
            stock_daily_return.rolling(63).std() * np.sqrt(252)
        ).resample("M").last()

        rolling_cov = stock_daily_return.rolling(63).cov(qqq_daily_return)
        rolling_var = qqq_daily_return.rolling(63).var()

        df["beta_3m"] = (rolling_cov / rolling_var).resample("M").last()
        df["corr_qqq_3m"] = (
            stock_daily_return.rolling(63).corr(qqq_daily_return)
        ).resample("M").last()

        df["tracking_error_3m"] = (
            (stock_daily_return - qqq_daily_return).rolling(63).std() * np.sqrt(252)
        ).resample("M").last()

        df["max_drawdown_3m"] = (
            stock_daily_return.rolling(63).apply(max_drawdown, raw=False)
        ).resample("M").last()

        # QQQ-relative return features
        df["relative_return_1m"] = df["ret_1m"] - qqq_monthly_return
        df["relative_return_3m"] = df["ret_3m"] - monthly_close[benchmark].pct_change(3, fill_method=None)
        df["relative_return_6m"] = df["ret_6m"] - monthly_close[benchmark].pct_change(6, fill_method=None)
        df["relative_return_12m"] = df["ret_12m"] - monthly_close[benchmark].pct_change(12, fill_method=None)

        # Future returns used only for labelling and backtesting
        df["future_stock_return"] = monthly_returns[ticker].shift(-1)
        df["future_qqq_return"] = qqq_monthly_return.shift(-1)

        df[TARGET_COL] = (
            df["future_stock_return"] > df["future_qqq_return"]
        ).astype(int)

        feature_frames.append(df.reset_index().rename(columns={"index": "date"}))

    panel = pd.concat(feature_frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])

    feature_cols = [
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "volume_growth_3m",
        "volatility_1m",
        "volatility_3m",
        "beta_3m",
        "corr_qqq_3m",
        "tracking_error_3m",
        "max_drawdown_3m",
        "relative_return_1m",
        "relative_return_3m",
        "relative_return_6m",
        "relative_return_12m",
    ]

    before_cleaning = len(panel)

    panel = panel.dropna(
        subset=feature_cols + ["future_stock_return", "future_qqq_return", TARGET_COL]
    ).copy()

    after_cleaning = len(panel)
    rows_removed = before_cleaning - after_cleaning

    panel = panel[
        ["date", "ticker"]
        + feature_cols
        + ["future_stock_return", "future_qqq_return", TARGET_COL]
    ]

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    return panel, feature_cols, before_cleaning, after_cleaning, rows_removed


def save_dataset_summary(panel, feature_cols, before_cleaning, after_cleaning, rows_removed):
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "start_date": panel["date"].min(),
        "end_date": panel["date"].max(),
        "number_of_stocks": panel["ticker"].nunique(),
        "stock_month_observations_before_cleaning": before_cleaning,
        "stock_month_observations_after_cleaning": after_cleaning,
        "rows_removed_due_to_missing_values": rows_removed,
        "number_of_predictive_features": len(feature_cols),
        "target_positive_count": int((panel[TARGET_COL] == 1).sum()),
        "target_negative_count": int((panel[TARGET_COL] == 0).sum()),
        "target_positive_rate": float((panel[TARGET_COL] == 1).mean()),
        "target_negative_rate": float((panel[TARGET_COL] == 0).mean()),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(results_dir / "dataset_summary.csv", index=False)

    feature_df = pd.DataFrame({"feature": feature_cols})
    feature_df.to_csv(results_dir / "feature_list.csv", index=False)

    class_balance = (
        panel[TARGET_COL]
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_balance["percentage"] = class_balance["count"] / class_balance["count"].sum()
    class_balance.to_csv(results_dir / "class_balance.csv", index=False)

    with open(results_dir / "dataset_summary.txt", "w") as f:
        f.write("DATASET SUMMARY\n")
        f.write("================\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nFEATURE LIST\n")
        f.write("============\n")
        for feature in feature_cols:
            f.write(f"- {feature}\n")

    print("\nDataset summary saved to results/tables/dataset_summary.csv")
    print("Feature list saved to results/tables/feature_list.csv")
    print("Class balance saved to results/tables/class_balance.csv")


def main():
    config = load_config()

    benchmark = config.get("project", {}).get("benchmark", "QQQ")

    raw_prices_file = Path(
        config.get("data", {}).get("raw_prices_file", "data/raw/prices.csv")
    )

    model_panel_file = Path(
        config.get("data", {}).get("model_panel_file", "data/processed/model_panel.csv")
    )

    print(f"Loading raw prices from: {raw_prices_file}")
    prices = pd.read_csv(raw_prices_file)
    prices = standardise_price_columns(prices)

    print(f"Building monthly model panel using benchmark: {benchmark}")
    panel, feature_cols, before_cleaning, after_cleaning, rows_removed = build_model_panel(
        prices=prices,
        benchmark=benchmark,
    )

    model_panel_file.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(model_panel_file, index=False)

    save_dataset_summary(
        panel=panel,
        feature_cols=feature_cols,
        before_cleaning=before_cleaning,
        after_cleaning=after_cleaning,
        rows_removed=rows_removed,
    )

    print("\nFeature engineering complete.")
    print(f"Saved model panel to: {model_panel_file}")
    print(f"Rows before cleaning: {before_cleaning}")
    print(f"Rows after cleaning: {after_cleaning}")
    print(f"Rows removed: {rows_removed}")
    print(f"Number of stocks: {panel['ticker'].nunique()}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print("\nFirst five rows:")
    print(panel.head())


if __name__ == "__main__":
    main()