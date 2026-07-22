from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


TARGET_COL = "outperform_qqq_next_1m"


REQUIRED_OUTPUTS = [
    ("data/processed/model_panel.csv", "data"),
    ("results/tables/dataset_summary.csv", "table"),
    ("results/tables/class_balance.csv", "table"),
    ("results/tables/feature_list.csv", "table"),
    ("results/tables/model_metrics.csv", "table"),
    ("results/tables/selected_hyperparameters.csv", "table"),
    ("results/tables/optimisation_search_results.csv", "table"),
    ("results/tables/rf_predictions.csv", "table"),
    ("results/tables/portfolio_performance.csv", "table"),
    ("results/tables/benchmark_comparison.csv", "table"),
    ("results/tables/portfolio_monthly_returns.csv", "table"),
    ("results/tables/portfolio_holdings.csv", "table"),
    ("results/tables/transaction_cost_sensitivity.csv", "table"),
    ("results/tables/topk_robustness_results.csv", "table"),
    ("results/tables/ablation_classification_results.csv", "table"),
    ("results/tables/ablation_portfolio_results.csv", "table"),
    ("results/figures/confusion_matrix_baseline.png", "figure"),
    ("results/figures/confusion_matrix_optimised.png", "figure"),
    ("results/figures/roc_curve_comparison.png", "figure"),
    ("results/figures/cumulative_returns_comparison.png", "figure"),
    ("results/figures/drawdown_comparison.png", "figure"),
    ("results/figures/topk_robustness_total_return.png", "figure"),
    ("results/figures/ablation_total_return.png", "figure"),
    ("results/models/baseline_random_forest.joblib", "model"),
    ("results/models/optimised_random_forest.joblib", "model"),
]


def ensure_output_dirs():
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        print(f"Could not read {path}: {error}")
        return None


def identify_feature_columns(panel):
    excluded = {
        "date",
        "ticker",
        "future_stock_return",
        "future_qqq_return",
        TARGET_COL,
    }

    return [col for col in panel.columns if col not in excluded]


def create_file_audit():
    rows = []

    for file_path, category in REQUIRED_OUTPUTS:
        path = Path(file_path)

        if path.exists():
            status = "exists"
            size_bytes = path.stat().st_size
            modified_at = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            status = "missing"
            size_bytes = np.nan
            modified_at = ""

        rows.append(
            {
                "file_path": file_path,
                "category": category,
                "status": status,
                "size_bytes": size_bytes,
                "modified_at": modified_at,
            }
        )

    return pd.DataFrame(rows)


def summarise_dataset(panel):
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    feature_cols = identify_feature_columns(panel)

    positive_count = int(panel[TARGET_COL].sum())
    total_count = int(len(panel))
    negative_count = total_count - positive_count

    positive_rate = positive_count / total_count
    negative_rate = negative_count / total_count

    summary = {
        "benchmark": "QQQ",
        "number_of_stock_month_observations": total_count,
        "number_of_unique_stocks": int(panel["ticker"].nunique()),
        "sample_start_date": str(panel["date"].min().date()),
        "sample_end_date": str(panel["date"].max().date()),
        "number_of_predictive_features": int(len(feature_cols)),
        "positive_class_count": positive_count,
        "negative_class_count": negative_count,
        "positive_class_percentage": positive_rate,
        "negative_class_percentage": negative_rate,
        "target_variable": TARGET_COL,
    }

    return summary


def best_model_row(model_metrics):
    if model_metrics is None or model_metrics.empty:
        return None

    if "roc_auc" in model_metrics.columns:
        return model_metrics.sort_values("roc_auc", ascending=False).iloc[0].to_dict()

    if "accuracy" in model_metrics.columns:
        return model_metrics.sort_values("accuracy", ascending=False).iloc[0].to_dict()

    return model_metrics.iloc[0].to_dict()


def best_portfolio_row(portfolio_performance):
    if portfolio_performance is None or portfolio_performance.empty:
        return None

    data = portfolio_performance.copy()

    if "return_type" in data.columns:
        net_rows = data[
            data["return_type"].astype(str).str.contains("net", case=False, na=False)
        ].copy()

        if len(net_rows) > 0:
            data = net_rows

    if "total_return" in data.columns:
        return data.sort_values("total_return", ascending=False).iloc[0].to_dict()

    if "sharpe_ratio" in data.columns:
        return data.sort_values("sharpe_ratio", ascending=False).iloc[0].to_dict()

    return data.iloc[0].to_dict()


def extract_named_strategy(portfolio_performance, strategy_name):
    if portfolio_performance is None or portfolio_performance.empty:
        return None

    if "strategy" not in portfolio_performance.columns:
        return None

    data = portfolio_performance[
        portfolio_performance["strategy"].astype(str) == strategy_name
    ].copy()

    if data.empty:
        return None

    if "return_type" in data.columns:
        net_data = data[
            data["return_type"].astype(str).str.contains("net", case=False, na=False)
        ].copy()

        if len(net_data) > 0:
            data = net_data

    return data.iloc[0].to_dict()


def calculate_difference(value_a, value_b):
    try:
        return float(value_a) - float(value_b)
    except Exception:
        return np.nan


def create_key_results_summary(
    dataset_summary,
    model_metrics,
    portfolio_performance,
    topk_results,
    ablation_classification,
    ablation_portfolio,
):
    rows = []

    for key, value in dataset_summary.items():
        rows.append(
            {
                "section": "dataset",
                "metric": key,
                "value": value,
                "interpretation": "Dataset characteristic used in Chapter 3 and Chapter 6.",
            }
        )

    best_model = best_model_row(model_metrics)

    if best_model is not None:
        model_name = best_model.get("model", "unknown_model")

        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if metric in best_model:
                rows.append(
                    {
                        "section": "classification",
                        "metric": f"best_model_{metric}",
                        "value": best_model[metric],
                        "interpretation": f"Best observed classification metric for {model_name}.",
                    }
                )

    best_portfolio = best_portfolio_row(portfolio_performance)

    if best_portfolio is not None:
        strategy_name = best_portfolio.get("strategy", "unknown_strategy")

        for metric in [
            "total_return",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "average_monthly_turnover",
        ]:
            if metric in best_portfolio:
                rows.append(
                    {
                        "section": "portfolio",
                        "metric": f"best_portfolio_{metric}",
                        "value": best_portfolio[metric],
                        "interpretation": f"Best portfolio result based on available summary table: {strategy_name}.",
                    }
                )

    optimised = extract_named_strategy(portfolio_performance, "optimised_rf_top_k")
    baseline = extract_named_strategy(portfolio_performance, "baseline_rf_top_k")
    qqq = extract_named_strategy(portfolio_performance, "qqq_buy_hold")
    equal_weight = extract_named_strategy(portfolio_performance, "equal_weight_universe")

    if optimised is not None and baseline is not None:
        rows.append(
            {
                "section": "portfolio_comparison",
                "metric": "optimised_minus_baseline_total_return",
                "value": calculate_difference(
                    optimised.get("total_return"), baseline.get("total_return")
                ),
                "interpretation": "Difference between optimised Random Forest Top-K total return and baseline Random Forest Top-K total return.",
            }
        )

    if optimised is not None and qqq is not None:
        rows.append(
            {
                "section": "portfolio_comparison",
                "metric": "optimised_minus_qqq_total_return",
                "value": calculate_difference(
                    optimised.get("total_return"), qqq.get("total_return")
                ),
                "interpretation": "Difference between optimised Random Forest Top-K total return and QQQ buy-and-hold total return.",
            }
        )

    if optimised is not None and equal_weight is not None:
        rows.append(
            {
                "section": "portfolio_comparison",
                "metric": "optimised_minus_equal_weight_total_return",
                "value": calculate_difference(
                    optimised.get("total_return"), equal_weight.get("total_return")
                ),
                "interpretation": "Difference between optimised Random Forest Top-K total return and equal-weight universe total return.",
            }
        )

    if topk_results is not None and not topk_results.empty:
        if "total_return" in topk_results.columns:
            best_topk = topk_results.sort_values("total_return", ascending=False).iloc[0]
            rows.append(
                {
                    "section": "robustness",
                    "metric": "best_topk_configuration",
                    "value": f"{best_topk.get('strategy', '')}, Top-{best_topk.get('top_k', '')}",
                    "interpretation": "Best Top-K robustness configuration by total return.",
                }
            )
            rows.append(
                {
                    "section": "robustness",
                    "metric": "best_topk_total_return",
                    "value": best_topk.get("total_return", np.nan),
                    "interpretation": "Total return of the best Top-K robustness configuration.",
                }
            )

    if ablation_portfolio is not None and not ablation_portfolio.empty:
        if "total_return" in ablation_portfolio.columns:
            best_ablation = ablation_portfolio.sort_values(
                "total_return", ascending=False
            ).iloc[0]

            rows.append(
                {
                    "section": "ablation",
                    "metric": "best_ablation_portfolio_model",
                    "value": best_ablation.get("model", ""),
                    "interpretation": "Best ablation model by portfolio total return.",
                }
            )
            rows.append(
                {
                    "section": "ablation",
                    "metric": "best_ablation_total_return",
                    "value": best_ablation.get("total_return", np.nan),
                    "interpretation": "Portfolio total return of the best ablation model.",
                }
            )

    if ablation_classification is not None and not ablation_classification.empty:
        if "roc_auc" in ablation_classification.columns:
            best_ablation_classification = ablation_classification.sort_values(
                "roc_auc", ascending=False
            ).iloc[0]

            rows.append(
                {
                    "section": "ablation",
                    "metric": "best_ablation_classification_model",
                    "value": best_ablation_classification.get("model", ""),
                    "interpretation": "Best ablation model by ROC-AUC.",
                }
            )
            rows.append(
                {
                    "section": "ablation",
                    "metric": "best_ablation_roc_auc",
                    "value": best_ablation_classification.get("roc_auc", np.nan),
                    "interpretation": "ROC-AUC of the best ablation classification model.",
                }
            )

    rows.append(
        {
            "section": "implementation_scope",
            "metric": "implemented_explainability_method",
            "value": "Random Forest feature importance and feature-group ablation; SHAP not implemented in final code.",
            "interpretation": "Use this to keep the dissertation aligned with the supervisor instruction to avoid SHAP for now.",
        }
    )

    return pd.DataFrame(rows)


def format_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "N/A"

        if abs(value) < 1:
            return f"{value:.4f}"

        return f"{value:.4f}"

    return str(value)


def create_text_summary(key_results, audit):
    lines = []

    lines.append("FINAL IMPLEMENTATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    total_files = len(audit)
    existing_files = int((audit["status"] == "exists").sum())
    missing_files = total_files - existing_files

    lines.append("1. Output-file audit")
    lines.append("-" * 80)
    lines.append(f"Required files checked: {total_files}")
    lines.append(f"Files found: {existing_files}")
    lines.append(f"Files missing: {missing_files}")
    lines.append("")

    if missing_files > 0:
        lines.append("Missing files:")
        missing = audit[audit["status"] == "missing"]["file_path"].tolist()
        for file_path in missing:
            lines.append(f"- {file_path}")
        lines.append("")

    lines.append("2. Key results for dissertation")
    lines.append("-" * 80)

    for _, row in key_results.iterrows():
        section = row["section"]
        metric = row["metric"]
        value = format_value(row["value"])
        interpretation = row["interpretation"]

        lines.append(f"[{section}] {metric}: {value}")
        lines.append(f"Interpretation: {interpretation}")
        lines.append("")

    lines.append("3. Report-writing guidance")
    lines.append("-" * 80)
    lines.append(
        "Use the dataset values in Chapter 3, the model metrics in Chapter 6, "
        "the portfolio and benchmark comparisons in Chapter 6, and the robustness "
        "and ablation results in Chapter 6 and Chapter 7."
    )
    lines.append("")
    lines.append(
        "The final report should not claim that SHAP was implemented unless SHAP code "
        "and SHAP outputs are actually added later. For the current implementation, "
        "the appropriate explanation/diagnostic evidence is Random Forest feature "
        "importance and feature-group ablation."
    )
    lines.append("")
    lines.append(
        "Recommended final technical claim: the implementation provides an end-to-end "
        "time-aware Random Forest framework for QQQ-relative stock ranking, Top-K "
        "portfolio construction, benchmark comparison, transaction-cost testing, "
        "Top-K robustness testing and feature-group ablation."
    )

    return "\n".join(lines)


def main():
    ensure_output_dirs()

    print("Generating final implementation audit...")

    audit = create_file_audit()
    audit.to_csv("results/tables/final_implementation_audit.csv", index=False)

    panel = read_csv_if_exists("data/processed/model_panel.csv")

    if panel is None:
        raise FileNotFoundError(
            "data/processed/model_panel.csv was not found. Run scripts/02_build_features.py first."
        )

    dataset_summary = summarise_dataset(panel)

    model_metrics = read_csv_if_exists("results/tables/model_metrics.csv")
    portfolio_performance = read_csv_if_exists("results/tables/portfolio_performance.csv")
    topk_results = read_csv_if_exists("results/tables/topk_robustness_results.csv")
    ablation_classification = read_csv_if_exists(
        "results/tables/ablation_classification_results.csv"
    )
    ablation_portfolio = read_csv_if_exists(
        "results/tables/ablation_portfolio_results.csv"
    )

    key_results = create_key_results_summary(
        dataset_summary=dataset_summary,
        model_metrics=model_metrics,
        portfolio_performance=portfolio_performance,
        topk_results=topk_results,
        ablation_classification=ablation_classification,
        ablation_portfolio=ablation_portfolio,
    )

    key_results.to_csv("results/tables/final_key_results_summary.csv", index=False)

    text_summary = create_text_summary(key_results=key_results, audit=audit)

    with open("results/logs/final_results_summary.txt", "w", encoding="utf-8") as f:
        f.write(text_summary)

    print("\nFinal output generation complete.")
    print("Saved:")
    print("- results/tables/final_implementation_audit.csv")
    print("- results/tables/final_key_results_summary.csv")
    print("- results/logs/final_results_summary.txt")

    print("\nAudit summary:")
    print(audit[["file_path", "category", "status"]])

    print("\nKey results summary:")
    print(key_results)


if __name__ == "__main__":
    main()