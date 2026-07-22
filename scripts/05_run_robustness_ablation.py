from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


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


def identify_feature_columns(panel):
    excluded = {
        "date",
        "ticker",
        "future_stock_return",
        "future_qqq_return",
        TARGET_COL,
    }
    return [col for col in panel.columns if col not in excluded]


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def calculate_classification_metrics(y_true, y_proba, model_name):
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "positive_prediction_rate": float(np.mean(y_pred)),
    }


def calculate_turnover(previous_weights, current_weights):
    all_tickers = set(previous_weights.keys()).union(set(current_weights.keys()))
    turnover = 0.0

    for ticker in all_tickers:
        turnover += abs(current_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))

    return turnover


def run_top_k_backtest(predictions, probability_col, strategy_name, top_k=10, transaction_cost_bps=10):
    monthly_rows = []
    previous_weights = {}

    for date, group in predictions.groupby("date"):
        group = group.copy().sort_values(probability_col, ascending=False)
        selected = group.head(top_k).copy()

        if len(selected) == 0:
            continue

        selected_tickers = selected["ticker"].tolist()
        equal_weight = 1.0 / len(selected_tickers)

        current_weights = {ticker: equal_weight for ticker in selected_tickers}

        gross_return = float((selected["future_stock_return"] * equal_weight).sum())
        turnover = calculate_turnover(previous_weights, current_weights)
        transaction_cost = turnover * (transaction_cost_bps / 10000.0)
        net_return = gross_return - transaction_cost

        monthly_rows.append(
            {
                "date": date,
                "strategy": strategy_name,
                "top_k": top_k,
                "gross_return": gross_return,
                "net_return": net_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "number_of_holdings": len(selected_tickers),
            }
        )

        previous_weights = current_weights

    return pd.DataFrame(monthly_rows)


def calculate_performance_metrics(return_series, turnover_series=None, periods_per_year=12):
    returns = pd.Series(return_series).dropna()

    if len(returns) == 0:
        return {
            "total_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "average_monthly_turnover": np.nan,
            "positive_month_rate": np.nan,
        }

    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annualised_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    annualised_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)

    if annualised_volatility != 0 and not np.isnan(annualised_volatility):
        sharpe_ratio = annualised_return / annualised_volatility
    else:
        sharpe_ratio = np.nan

    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    if turnover_series is not None:
        average_turnover = pd.Series(turnover_series).dropna().mean()
    else:
        average_turnover = np.nan

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "average_monthly_turnover": average_turnover,
        "positive_month_rate": (returns > 0).mean(),
    }


def create_train_test_split(panel, test_months=24):
    unique_dates = pd.Series(panel["date"].sort_values().unique())
    n_dates = len(unique_dates)

    if n_dates <= test_months + 36:
        test_months = max(6, int(n_dates * 0.20))

    train_val_dates = unique_dates.iloc[:-test_months]
    test_dates = unique_dates.iloc[-test_months:]

    train_val = panel[panel["date"].isin(train_val_dates)].copy()
    test = panel[panel["date"].isin(test_dates)].copy()

    return train_val, test


def parse_selected_hyperparameters(path="results/tables/selected_hyperparameters.csv"):
    selected = pd.read_csv(path).iloc[0].to_dict()

    def parse_none(value):
        if pd.isna(value):
            return None
        if str(value).lower() in ["none", "nan", ""]:
            return None
        return value

    max_depth = parse_none(selected.get("max_depth"))
    class_weight = parse_none(selected.get("class_weight"))
    max_features = parse_none(selected.get("max_features"))

    if max_depth is not None:
        max_depth = int(float(max_depth))

    if max_features is not None:
        try:
            max_features = float(max_features)
        except ValueError:
            max_features = str(max_features)

    params = {
        "n_estimators": int(float(selected.get("n_estimators", 300))),
        "max_depth": max_depth,
        "min_samples_leaf": int(float(selected.get("min_samples_leaf", 5))),
        "max_features": max_features,
        "class_weight": class_weight,
        "random_state": int(float(selected.get("random_state", 42))),
        "n_jobs": -1,
    }

    return params


def get_feature_groups():
    return {
        "momentum": [
            "ret_1m",
            "ret_3m",
            "ret_6m",
            "ret_12m",
        ],
        "volume": [
            "volume_growth_3m",
        ],
        "risk": [
            "volatility_1m",
            "volatility_3m",
            "beta_3m",
            "corr_qqq_3m",
            "tracking_error_3m",
            "max_drawdown_3m",
        ],
        "qqq_relative": [
            "relative_return_1m",
            "relative_return_3m",
            "relative_return_6m",
            "relative_return_12m",
        ],
    }


def build_ablation_feature_sets(all_features):
    feature_groups = get_feature_groups()

    ablation_sets = {
        "full_optimised_model": all_features,
    }

    for group_name, group_features in feature_groups.items():
        remaining_features = [f for f in all_features if f not in group_features]
        ablation_sets[f"without_{group_name}"] = remaining_features

    return ablation_sets


def run_ablation_models(panel, params, test_months, top_k, transaction_cost_bps):
    all_features = identify_feature_columns(panel)
    ablation_feature_sets = build_ablation_feature_sets(all_features)

    train_val, test = create_train_test_split(panel, test_months=test_months)

    classification_rows = []
    portfolio_rows = []
    prediction_frames = []

    for ablation_name, feature_cols in ablation_feature_sets.items():
        print(f"Running ablation model: {ablation_name}")
        print(f"Number of features: {len(feature_cols)}")

        x_train = train_val[feature_cols]
        y_train = train_val[TARGET_COL]

        x_test = test[feature_cols]
        y_test = test[TARGET_COL]

        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)

        y_proba = model.predict_proba(x_test)[:, list(model.classes_).index(1)]

        classification_metrics = calculate_classification_metrics(
            y_true=y_test,
            y_proba=y_proba,
            model_name=ablation_name,
        )

        classification_metrics["number_of_features"] = len(feature_cols)
        classification_metrics["removed_feature_group"] = ablation_name.replace("without_", "")
        classification_rows.append(classification_metrics)

        prediction_frame = test[
            [
                "date",
                "ticker",
                "future_stock_return",
                "future_qqq_return",
                TARGET_COL,
            ]
        ].copy()

        prediction_frame["ablation_model"] = ablation_name
        prediction_frame["predicted_probability"] = y_proba
        prediction_frames.append(prediction_frame)

        monthly_returns = run_top_k_backtest(
            predictions=prediction_frame,
            probability_col="predicted_probability",
            strategy_name=ablation_name,
            top_k=top_k,
            transaction_cost_bps=transaction_cost_bps,
        )

        performance = calculate_performance_metrics(
            return_series=monthly_returns["net_return"],
            turnover_series=monthly_returns["turnover"],
        )

        performance["model"] = ablation_name
        performance["top_k"] = top_k
        performance["number_of_features"] = len(feature_cols)
        portfolio_rows.append(performance)

    classification_results = pd.DataFrame(classification_rows)
    portfolio_results = pd.DataFrame(portfolio_rows)
    ablation_predictions = pd.concat(prediction_frames, ignore_index=True)

    return classification_results, portfolio_results, ablation_predictions


def run_top_k_robustness(predictions, top_k_values, transaction_cost_bps):
    rows = []

    strategy_specs = [
        ("baseline_rf_top_k", "baseline_predicted_probability"),
        ("optimised_rf_top_k", "optimised_predicted_probability"),
    ]

    for top_k in top_k_values:
        for strategy_name, probability_col in strategy_specs:
            monthly_returns = run_top_k_backtest(
                predictions=predictions,
                probability_col=probability_col,
                strategy_name=strategy_name,
                top_k=top_k,
                transaction_cost_bps=transaction_cost_bps,
            )

            performance = calculate_performance_metrics(
                return_series=monthly_returns["net_return"],
                turnover_series=monthly_returns["turnover"],
            )

            performance["strategy"] = strategy_name
            performance["top_k"] = top_k
            rows.append(performance)

    return pd.DataFrame(rows)


def save_topk_figure(topk_results, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for strategy, group in topk_results.groupby("strategy"):
        group = group.sort_values("top_k")
        ax.plot(group["top_k"], group["total_return"], marker="o", label=strategy)

    ax.set_title("Top-K Robustness: Total Return")
    ax.set_xlabel("Top-K holdings")
    ax.set_ylabel("Total return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_ablation_figure(ablation_portfolio_results, output_path):
    plot_data = ablation_portfolio_results.sort_values("total_return", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(plot_data["model"], plot_data["total_return"])
    ax.set_title("Feature-Group Ablation: Portfolio Total Return")
    ax.set_xlabel("Model")
    ax.set_ylabel("Total return")
    ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    ensure_output_dirs()

    config = load_config()

    model_panel_file = Path(
        get_nested(config, ["data", "model_panel_file"], "data/processed/model_panel.csv")
    )

    predictions_file = Path("results/tables/rf_predictions.csv")

    test_months = int(get_nested(config, ["validation", "test_months"], 24))
    top_k = int(get_nested(config, ["portfolio", "top_k"], 10))
    transaction_cost_bps = float(get_nested(config, ["portfolio", "transaction_cost_bps"], 10))

    top_k_values = get_nested(config, ["robustness", "top_k_values"], [5, 10, 15, 20])

    print(f"Loading model panel from: {model_panel_file}")
    panel = pd.read_csv(model_panel_file)
    panel["date"] = pd.to_datetime(panel["date"])

    feature_cols = identify_feature_columns(panel)
    panel = panel.dropna(subset=feature_cols + [TARGET_COL]).copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Loading Random Forest predictions from: {predictions_file}")
    predictions = pd.read_csv(predictions_file)
    predictions["date"] = pd.to_datetime(predictions["date"])

    print("Loading selected optimised hyperparameters...")
    params = parse_selected_hyperparameters()

    print("\nRunning Top-K robustness checks...")
    topk_results = run_top_k_robustness(
        predictions=predictions,
        top_k_values=top_k_values,
        transaction_cost_bps=transaction_cost_bps,
    )

    print("\nRunning feature-group ablation tests...")
    ablation_classification_results, ablation_portfolio_results, ablation_predictions = run_ablation_models(
        panel=panel,
        params=params,
        test_months=test_months,
        top_k=top_k,
        transaction_cost_bps=transaction_cost_bps,
    )

    topk_results.to_csv("results/tables/topk_robustness_results.csv", index=False)
    ablation_classification_results.to_csv(
        "results/tables/ablation_classification_results.csv",
        index=False,
    )
    ablation_portfolio_results.to_csv(
        "results/tables/ablation_portfolio_results.csv",
        index=False,
    )
    ablation_predictions.to_csv(
        "results/tables/ablation_predictions.csv",
        index=False,
    )

    save_topk_figure(
        topk_results=topk_results,
        output_path="results/figures/topk_robustness_total_return.png",
    )

    save_ablation_figure(
        ablation_portfolio_results=ablation_portfolio_results,
        output_path="results/figures/ablation_total_return.png",
    )

    print("\nRobustness and ablation tests complete.")
    print("Saved outputs:")
    print("- results/tables/topk_robustness_results.csv")
    print("- results/tables/ablation_classification_results.csv")
    print("- results/tables/ablation_portfolio_results.csv")
    print("- results/tables/ablation_predictions.csv")
    print("- results/figures/topk_robustness_total_return.png")
    print("- results/figures/ablation_total_return.png")

    print("\nTop-K robustness results:")
    print(topk_results)

    print("\nAblation classification results:")
    print(ablation_classification_results)

    print("\nAblation portfolio results:")
    print(ablation_portfolio_results)


if __name__ == "__main__":
    main()