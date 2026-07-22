from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TARGET_COL = "outperform_qqq_next_1m"


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_nested(config, key_path, default=None):
    current = config
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def ensure_output_dirs():
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)


def calculate_turnover(previous_weights, current_weights):
    all_tickers = set(previous_weights.keys()).union(set(current_weights.keys()))
    turnover = 0.0

    for ticker in all_tickers:
        previous_weight = previous_weights.get(ticker, 0.0)
        current_weight = current_weights.get(ticker, 0.0)
        turnover += abs(current_weight - previous_weight)

    return turnover


def run_top_k_backtest(
    predictions,
    probability_col,
    strategy_name,
    top_k=10,
    transaction_cost_bps=10,
):
    monthly_rows = []
    holdings_rows = []

    previous_weights = {}

    for date, group in predictions.groupby("date"):
        group = group.copy()
        group = group.sort_values(probability_col, ascending=False)

        selected = group.head(top_k).copy()

        if len(selected) == 0:
            continue

        selected_tickers = selected["ticker"].tolist()
        equal_weight = 1.0 / len(selected_tickers)

        current_weights = {ticker: equal_weight for ticker in selected_tickers}

        gross_return = float((selected["future_stock_return"] * equal_weight).sum())

        turnover = calculate_turnover(
            previous_weights=previous_weights,
            current_weights=current_weights,
        )

        transaction_cost = turnover * (transaction_cost_bps / 10000.0)
        net_return = gross_return - transaction_cost

        monthly_rows.append(
            {
                "date": date,
                "strategy": strategy_name,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "turnover": turnover,
                "number_of_holdings": len(selected_tickers),
                "average_predicted_probability": selected[probability_col].mean(),
            }
        )

        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            holdings_rows.append(
                {
                    "date": date,
                    "strategy": strategy_name,
                    "rank": rank,
                    "ticker": row["ticker"],
                    "weight": equal_weight,
                    "predicted_probability": row[probability_col],
                    "future_stock_return": row["future_stock_return"],
                    "future_qqq_return": row["future_qqq_return"],
                    "outperformed_qqq": row[TARGET_COL],
                }
            )

        previous_weights = current_weights

    monthly_returns = pd.DataFrame(monthly_rows)
    holdings = pd.DataFrame(holdings_rows)

    return monthly_returns, holdings


def calculate_performance_metrics(return_series, turnover_series=None, periods_per_year=12):
    returns = pd.Series(return_series).dropna()

    if len(returns) == 0:
        return {
            "total_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
            "average_monthly_return": np.nan,
            "positive_month_rate": np.nan,
            "average_monthly_turnover": np.nan,
        }

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    annualised_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    annualised_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)

    if annualised_volatility != 0 and not np.isnan(annualised_volatility):
        sharpe_ratio = annualised_return / annualised_volatility
    else:
        sharpe_ratio = np.nan

    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std(ddof=1) * np.sqrt(periods_per_year)

    if downside_volatility != 0 and not np.isnan(downside_volatility):
        sortino_ratio = annualised_return / downside_volatility
    else:
        sortino_ratio = np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    if max_drawdown != 0 and not np.isnan(max_drawdown):
        calmar_ratio = annualised_return / abs(max_drawdown)
    else:
        calmar_ratio = np.nan

    if turnover_series is not None:
        average_turnover = pd.Series(turnover_series).dropna().mean()
    else:
        average_turnover = np.nan

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "average_monthly_return": returns.mean(),
        "positive_month_rate": (returns > 0).mean(),
        "average_monthly_turnover": average_turnover,
    }


def build_benchmarks(predictions):
    benchmark_rows = []

    for date, group in predictions.groupby("date"):
        qqq_return = group["future_qqq_return"].iloc[0]
        equal_weight_return = group["future_stock_return"].mean()

        benchmark_rows.append(
            {
                "date": date,
                "qqq_buy_hold": qqq_return,
                "equal_weight_universe": equal_weight_return,
            }
        )

    return pd.DataFrame(benchmark_rows).sort_values("date")


def create_performance_summary(monthly_returns, benchmarks):
    rows = []

    for strategy, group in monthly_returns.groupby("strategy"):
        net_metrics = calculate_performance_metrics(
            return_series=group["net_return"],
            turnover_series=group["turnover"],
        )
        net_metrics["strategy"] = strategy
        net_metrics["return_type"] = "net_after_transaction_costs"
        rows.append(net_metrics)

        gross_metrics = calculate_performance_metrics(
            return_series=group["gross_return"],
            turnover_series=group["turnover"],
        )
        gross_metrics["strategy"] = strategy
        gross_metrics["return_type"] = "gross_before_transaction_costs"
        rows.append(gross_metrics)

    qqq_metrics = calculate_performance_metrics(benchmarks["qqq_buy_hold"])
    qqq_metrics["strategy"] = "qqq_buy_hold"
    qqq_metrics["return_type"] = "benchmark"
    rows.append(qqq_metrics)

    equal_weight_metrics = calculate_performance_metrics(
        benchmarks["equal_weight_universe"]
    )
    equal_weight_metrics["strategy"] = "equal_weight_universe"
    equal_weight_metrics["return_type"] = "benchmark"
    rows.append(equal_weight_metrics)

    summary = pd.DataFrame(rows)

    ordered_cols = [
        "strategy",
        "return_type",
        "total_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "average_monthly_return",
        "positive_month_rate",
        "average_monthly_turnover",
    ]

    return summary[ordered_cols]


def create_wide_monthly_returns(monthly_returns, benchmarks):
    baseline = monthly_returns[
        monthly_returns["strategy"] == "baseline_rf_top_k"
    ][["date", "gross_return", "net_return", "turnover"]].copy()

    baseline = baseline.rename(
        columns={
            "gross_return": "baseline_rf_gross_return",
            "net_return": "baseline_rf_net_return",
            "turnover": "baseline_rf_turnover",
        }
    )

    optimised = monthly_returns[
        monthly_returns["strategy"] == "optimised_rf_top_k"
    ][["date", "gross_return", "net_return", "turnover"]].copy()

    optimised = optimised.rename(
        columns={
            "gross_return": "optimised_rf_gross_return",
            "net_return": "optimised_rf_net_return",
            "turnover": "optimised_rf_turnover",
        }
    )

    wide = benchmarks.merge(baseline, on="date", how="left")
    wide = wide.merge(optimised, on="date", how="left")

    return wide.sort_values("date")


def save_cumulative_return_figure(wide_returns, output_path):
    plot_data = wide_returns.copy()

    columns = {
        "qqq_buy_hold": "QQQ Buy-and-Hold",
        "equal_weight_universe": "Equal-Weight Universe",
        "baseline_rf_net_return": "Baseline RF Top-K",
        "optimised_rf_net_return": "Optimised RF Top-K",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label in columns.items():
        if col in plot_data.columns:
            cumulative = (1 + plot_data[col].fillna(0)).cumprod()
            ax.plot(plot_data["date"], cumulative, label=label)

    ax.set_title("Cumulative Return Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_drawdown_figure(wide_returns, output_path):
    plot_data = wide_returns.copy()

    columns = {
        "qqq_buy_hold": "QQQ Buy-and-Hold",
        "equal_weight_universe": "Equal-Weight Universe",
        "baseline_rf_net_return": "Baseline RF Top-K",
        "optimised_rf_net_return": "Optimised RF Top-K",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for col, label in columns.items():
        if col in plot_data.columns:
            cumulative = (1 + plot_data[col].fillna(0)).cumprod()
            drawdown = cumulative / cumulative.cummax() - 1
            ax.plot(plot_data["date"], drawdown, label=label)

    ax.set_title("Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_transaction_cost_sensitivity(predictions, top_k, cost_values):
    rows = []

    strategy_specs = [
        ("baseline_rf_top_k", "baseline_predicted_probability"),
        ("optimised_rf_top_k", "optimised_predicted_probability"),
    ]

    for cost_bps in cost_values:
        for strategy_name, probability_col in strategy_specs:
            monthly_returns, _ = run_top_k_backtest(
                predictions=predictions,
                probability_col=probability_col,
                strategy_name=strategy_name,
                top_k=top_k,
                transaction_cost_bps=cost_bps,
            )

            metrics = calculate_performance_metrics(
                return_series=monthly_returns["net_return"],
                turnover_series=monthly_returns["turnover"],
            )

            metrics["strategy"] = strategy_name
            metrics["transaction_cost_bps"] = cost_bps

            rows.append(metrics)

    return pd.DataFrame(rows)


def main():
    ensure_output_dirs()

    config = load_config()

    predictions_file = Path("results/tables/rf_predictions.csv")

    top_k = int(get_nested(config, ["portfolio", "top_k"], 10))

    transaction_cost_bps = float(
        get_nested(config, ["portfolio", "transaction_cost_bps"], 10)
    )

    cost_values = get_nested(
        config,
        ["portfolio", "transaction_cost_sensitivity_bps"],
        [0, 5, 10, 25, 50],
    )

    print(f"Loading predictions from: {predictions_file}")

    predictions = pd.read_csv(predictions_file)
    predictions["date"] = pd.to_datetime(predictions["date"])

    required_cols = [
        "date",
        "ticker",
        "future_stock_return",
        "future_qqq_return",
        TARGET_COL,
        "baseline_predicted_probability",
        "optimised_predicted_probability",
    ]

    missing_cols = [col for col in required_cols if col not in predictions.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in rf_predictions.csv: {missing_cols}")

    print(f"Rows in prediction file: {len(predictions)}")
    print(
        f"Date range: {predictions['date'].min().date()} "
        f"to {predictions['date'].max().date()}"
    )
    print(f"Top-K value: {top_k}")
    print(f"Transaction cost: {transaction_cost_bps} bps")

    baseline_returns, baseline_holdings = run_top_k_backtest(
        predictions=predictions,
        probability_col="baseline_predicted_probability",
        strategy_name="baseline_rf_top_k",
        top_k=top_k,
        transaction_cost_bps=transaction_cost_bps,
    )

    optimised_returns, optimised_holdings = run_top_k_backtest(
        predictions=predictions,
        probability_col="optimised_predicted_probability",
        strategy_name="optimised_rf_top_k",
        top_k=top_k,
        transaction_cost_bps=transaction_cost_bps,
    )

    monthly_returns = pd.concat(
        [baseline_returns, optimised_returns],
        ignore_index=True,
    ).sort_values(["date", "strategy"])

    holdings = pd.concat(
        [baseline_holdings, optimised_holdings],
        ignore_index=True,
    ).sort_values(["date", "strategy", "rank"])

    benchmarks = build_benchmarks(predictions)
    performance_summary = create_performance_summary(monthly_returns, benchmarks)
    wide_monthly_returns = create_wide_monthly_returns(monthly_returns, benchmarks)

    transaction_cost_sensitivity = run_transaction_cost_sensitivity(
        predictions=predictions,
        top_k=top_k,
        cost_values=cost_values,
    )

    monthly_returns.to_csv(
        "results/tables/portfolio_monthly_returns.csv",
        index=False,
    )

    holdings.to_csv(
        "results/tables/portfolio_holdings.csv",
        index=False,
    )

    benchmarks.to_csv(
        "results/tables/benchmark_monthly_returns.csv",
        index=False,
    )

    performance_summary.to_csv(
        "results/tables/portfolio_performance.csv",
        index=False,
    )

    performance_summary.to_csv(
        "results/tables/benchmark_comparison.csv",
        index=False,
    )

    wide_monthly_returns.to_csv(
        "results/tables/portfolio_comparison_monthly_returns.csv",
        index=False,
    )

    transaction_cost_sensitivity.to_csv(
        "results/tables/transaction_cost_sensitivity.csv",
        index=False,
    )

    save_cumulative_return_figure(
        wide_returns=wide_monthly_returns,
        output_path="results/figures/cumulative_returns_comparison.png",
    )

    save_drawdown_figure(
        wide_returns=wide_monthly_returns,
        output_path="results/figures/drawdown_comparison.png",
    )

    print("\nPortfolio backtest complete.")
    print("Saved outputs:")
    print("- results/tables/portfolio_monthly_returns.csv")
    print("- results/tables/portfolio_holdings.csv")
    print("- results/tables/benchmark_monthly_returns.csv")
    print("- results/tables/portfolio_performance.csv")
    print("- results/tables/benchmark_comparison.csv")
    print("- results/tables/portfolio_comparison_monthly_returns.csv")
    print("- results/tables/transaction_cost_sensitivity.csv")
    print("- results/figures/cumulative_returns_comparison.png")
    print("- results/figures/drawdown_comparison.png")

    print("\nPerformance summary:")
    print(performance_summary)


if __name__ == "__main__":
    main()