import itertools
import time
import warnings
from pathlib import Path

import joblib
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
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
    Path("results/models").mkdir(parents=True, exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)


def identify_feature_columns(panel):
    excluded = {
        "date",
        "ticker",
        "future_stock_return",
        "future_qqq_return",
        TARGET_COL,
    }
    return [col for col in panel.columns if col not in excluded]


def safe_predict_proba(model, x):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if 1 in model.classes_:
            class_index = list(model.classes_).index(1)
            return proba[:, class_index]
        return np.zeros(len(x))
    return model.predict(x)


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def calculate_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "positive_prediction_rate": float(np.mean(y_pred)),
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

    return train_val, test, train_val_dates, test_dates

def make_expanding_window_folds(
    train_val_dates,
    n_folds=4,
    min_train_months=36,
    validation_months=12,
    embargo_months=1,
):
    dates = pd.Series(pd.to_datetime(train_val_dates)).sort_values().reset_index(drop=True)
    n_dates = len(dates)

    if n_dates < min_train_months + validation_months + embargo_months:
        min_train_months = max(12, int(n_dates * 0.45))
        validation_months = max(3, int(n_dates * 0.10))

    earliest_validation_end = min_train_months + embargo_months + validation_months
    latest_validation_end = n_dates

    if earliest_validation_end >= latest_validation_end:
        raise ValueError(
            "Not enough dates to create expanding-window validation folds. "
            "Reduce min_train_months or validation_months."
        )

    validation_end_points = np.linspace(
        earliest_validation_end,
        latest_validation_end,
        num=n_folds + 1,
        dtype=int,
    )[1:]

    validation_end_points = sorted(set(validation_end_points))

    folds = []

    for fold_id, val_end in enumerate(validation_end_points, start=1):
        val_start = val_end - validation_months
        train_end = val_start - embargo_months

        if train_end < min_train_months:
            continue

        train_dates = dates.iloc[:train_end]
        validation_dates = dates.iloc[val_start:val_end]

        folds.append(
            {
                "fold": fold_id,
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "validation_start": validation_dates.min(),
                "validation_end": validation_dates.max(),
                "train_dates": train_dates,
                "validation_dates": validation_dates,
            }
        )

    if len(folds) == 0:
        raise ValueError("No valid validation folds were created.")

    return folds

def get_baseline_params(config):
    return {
        "n_estimators": int(get_nested(config, ["model", "n_estimators"], 300)),
        "max_depth": get_nested(config, ["model", "max_depth"], 8),
        "min_samples_leaf": int(get_nested(config, ["model", "min_samples_leaf"], 5)),
        "max_features": get_nested(config, ["model", "max_features"], "sqrt"),
        "class_weight": None,
        "random_state": int(get_nested(config, ["model", "random_state"], 42)),
        "n_jobs": -1,
    }


def build_parameter_candidates(random_state=42, n_iter=40):
    parameter_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced"],
    }

    keys = list(parameter_grid.keys())
    all_candidates = [
        dict(zip(keys, values))
        for values in itertools.product(*[parameter_grid[key] for key in keys])
    ]

    rng = np.random.default_rng(random_state)
    n_iter = min(n_iter, len(all_candidates))
    selected_indices = rng.choice(len(all_candidates), size=n_iter, replace=False)

    return [all_candidates[i] for i in selected_indices]


def evaluate_candidate(panel, feature_cols, target_col, folds, params, random_state=42):
    fold_results = []

    for fold in folds:
        train_data = panel[panel["date"].isin(fold["train_dates"])].copy()
        validation_data = panel[panel["date"].isin(fold["validation_dates"])].copy()

        x_train = train_data[feature_cols]
        y_train = train_data[target_col]

        x_validation = validation_data[feature_cols]
        y_validation = validation_data[target_col]

        model = RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)
        validation_proba = safe_predict_proba(model, x_validation)
        validation_auc = safe_roc_auc(y_validation, validation_proba)

        fold_results.append(
            {
                "fold": fold["fold"],
                "validation_roc_auc": validation_auc,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "train_rows": len(train_data),
                "validation_rows": len(validation_data),
            }
        )

    fold_auc_values = [result["validation_roc_auc"] for result in fold_results]
    mean_auc = np.nanmean(fold_auc_values)
    std_auc = np.nanstd(fold_auc_values)

    return mean_auc, std_auc, fold_results


def run_hyperparameter_search(
    train_val,
    feature_cols,
    target_col,
    folds,
    random_state=42,
    n_iter=40,
):
    candidates = build_parameter_candidates(
        random_state=random_state,
        n_iter=n_iter,
    )

    search_rows = []
    start_time = time.time()

    for candidate_id, params in enumerate(candidates, start=1):
        print(f"Evaluating candidate {candidate_id}/{len(candidates)}: {params}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_auc, std_auc, fold_results = evaluate_candidate(
                panel=train_val,
                feature_cols=feature_cols,
                target_col=target_col,
                folds=folds,
                params=params,
                random_state=random_state,
            )

        row = {
            "candidate_id": candidate_id,
            "mean_validation_roc_auc": mean_auc,
            "std_validation_roc_auc": std_auc,
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": params["max_features"],
            "class_weight": params["class_weight"],
        }

        search_rows.append(row)

    search_results = pd.DataFrame(search_rows)
    search_results = search_results.sort_values(
        by="mean_validation_roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    runtime_seconds = time.time() - start_time
    search_results["total_search_runtime_seconds"] = runtime_seconds

    best_row = search_results.iloc[0]

    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "max_features": best_row["max_features"],
        "class_weight": None if pd.isna(best_row["class_weight"]) else best_row["class_weight"],
        "random_state": random_state,
        "n_jobs": -1,
    }

    return search_results, best_params, runtime_seconds


def train_and_evaluate_final_model(
    train_val,
    test,
    feature_cols,
    target_col,
    params,
    model_name,
):
    x_train = train_val[feature_cols]
    y_train = train_val[target_col]

    x_test = test[feature_cols]
    y_test = test[target_col]

    model = RandomForestClassifier(**params)
    model.fit(x_train, y_train)

    test_proba = safe_predict_proba(model, x_test)
    test_pred = (test_proba >= 0.5).astype(int)

    metrics = calculate_metrics(y_test, test_proba)
    metrics["model"] = model_name
    metrics["train_rows"] = len(train_val)
    metrics["test_rows"] = len(test)

    return model, test_proba, test_pred, metrics


def save_confusion_matrix_figure(y_true, y_pred, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did not outperform QQQ", "Outperformed QQQ"],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_roc_comparison_figure(y_true, baseline_proba, optimised_proba, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        baseline_proba,
        name="Baseline Random Forest",
        ax=ax,
    )

    RocCurveDisplay.from_predictions(
        y_true,
        optimised_proba,
        name="Optimised Random Forest",
        ax=ax,
    )

    ax.plot([0, 1], [0, 1], linestyle="--", label="No-skill benchmark")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_feature_importance(model, feature_cols, output_path):
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(output_path, index=False)


def main():
    ensure_output_dirs()

    config = load_config()

    model_panel_file = Path(
        get_nested(config, ["data", "model_panel_file"], "data/processed/model_panel.csv")
    )

    random_state = int(get_nested(config, ["model", "random_state"], 42))

    test_months = int(
        get_nested(config, ["validation", "test_months"], 24)
    )

    n_folds = int(
        get_nested(config, ["validation", "n_folds"], 4)
    )

    min_train_months = int(
        get_nested(config, ["validation", "min_train_months"], 36)
    )

    validation_months = int(
        get_nested(config, ["validation", "validation_months"], 12)
    )

    embargo_months = int(
        get_nested(config, ["validation", "embargo_months"], 1)
    )

    n_iter = int(
        get_nested(config, ["optimisation", "n_iter"], 40)
    )

    print(f"Loading model panel from: {model_panel_file}")
    panel = pd.read_csv(model_panel_file)
    panel["date"] = pd.to_datetime(panel["date"])

    feature_cols = identify_feature_columns(panel)

    panel = panel.dropna(subset=feature_cols + [TARGET_COL]).copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Rows in final model panel: {len(panel)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"Number of stocks: {panel['ticker'].nunique()}")
    print("Target class balance:")
    print(panel[TARGET_COL].value_counts(normalize=True))

    train_val, test, train_val_dates, test_dates = create_train_test_split(
        panel=panel,
        test_months=test_months,
    )

    print("\nTime-aware split:")
    print(f"Training/validation period: {train_val['date'].min().date()} to {train_val['date'].max().date()}")
    print(f"Blind test period: {test['date'].min().date()} to {test['date'].max().date()}")
    print(f"Training/validation rows: {len(train_val)}")
    print(f"Blind test rows: {len(test)}")

    folds = make_expanding_window_folds(
        train_val_dates=train_val_dates,
        n_folds=n_folds,
        min_train_months=min_train_months,
        validation_months=validation_months,
        embargo_months=embargo_months,
    )

    fold_summary = pd.DataFrame(
        [
            {
                "fold": fold["fold"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "train_months": len(fold["train_dates"]),
                "validation_months": len(fold["validation_dates"]),
            }
            for fold in folds
        ]
    )

    fold_summary.to_csv("results/tables/time_aware_validation_folds.csv", index=False)

    print("\nValidation folds:")
    print(fold_summary)

    baseline_params = get_baseline_params(config)

    print("\nTraining baseline Random Forest...")
    baseline_model, baseline_proba, baseline_pred, baseline_metrics = train_and_evaluate_final_model(
        train_val=train_val,
        test=test,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        params=baseline_params,
        model_name="baseline_random_forest",
    )

    print("Baseline metrics:")
    print(baseline_metrics)

    print("\nRunning time-aware Randomized Search for optimised Random Forest...")
    search_results, best_params, runtime_seconds = run_hyperparameter_search(
        train_val=train_val,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        folds=folds,
        random_state=random_state,
        n_iter=n_iter,
    )

    print("\nBest optimised parameters:")
    print(best_params)

    print("\nTraining optimised Random Forest...")
    optimised_model, optimised_proba, optimised_pred, optimised_metrics = train_and_evaluate_final_model(
        train_val=train_val,
        test=test,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        params=best_params,
        model_name="optimised_random_forest",
    )

    print("Optimised metrics:")
    print(optimised_metrics)

    predictions = test[
        [
            "date",
            "ticker",
            "future_stock_return",
            "future_qqq_return",
            TARGET_COL,
        ]
    ].copy()

    predictions["baseline_predicted_probability"] = baseline_proba
    predictions["baseline_predicted_label"] = baseline_pred
    predictions["optimised_predicted_probability"] = optimised_proba
    predictions["optimised_predicted_label"] = optimised_pred

    predictions.to_csv("results/tables/rf_predictions.csv", index=False)

    model_metrics = pd.DataFrame([baseline_metrics, optimised_metrics])
    model_metrics = model_metrics[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "positive_prediction_rate",
            "train_rows",
            "test_rows",
        ]
    ]
    model_metrics.to_csv("results/tables/model_metrics.csv", index=False)

    search_results.to_csv("results/tables/optimisation_search_results.csv", index=False)

    selected_hyperparameters = pd.DataFrame([best_params])
    selected_hyperparameters["optimisation_runtime_seconds"] = runtime_seconds
    selected_hyperparameters.to_csv(
        "results/tables/selected_hyperparameters.csv",
        index=False,
    )

    joblib.dump(
        baseline_model,
        "results/models/baseline_random_forest.joblib",
    )

    joblib.dump(
        optimised_model,
        "results/models/optimised_random_forest.joblib",
    )

    save_confusion_matrix_figure(
        y_true=test[TARGET_COL],
        y_pred=baseline_pred,
        title="Baseline Random Forest Confusion Matrix",
        output_path="results/figures/confusion_matrix_baseline.png",
    )

    save_confusion_matrix_figure(
        y_true=test[TARGET_COL],
        y_pred=optimised_pred,
        title="Optimised Random Forest Confusion Matrix",
        output_path="results/figures/confusion_matrix_optimised.png",
    )

    save_roc_comparison_figure(
        y_true=test[TARGET_COL],
        baseline_proba=baseline_proba,
        optimised_proba=optimised_proba,
        output_path="results/figures/roc_curve_comparison.png",
    )

    save_feature_importance(
        model=optimised_model,
        feature_cols=feature_cols,
        output_path="results/tables/optimised_rf_feature_importance.csv",
    )

    print("\nModel training complete.")
    print("Saved outputs:")
    print("- results/tables/rf_predictions.csv")
    print("- results/tables/model_metrics.csv")
    print("- results/tables/optimisation_search_results.csv")
    print("- results/tables/selected_hyperparameters.csv")
    print("- results/tables/time_aware_validation_folds.csv")
    print("- results/tables/optimised_rf_feature_importance.csv")
    print("- results/models/baseline_random_forest.joblib")
    print("- results/models/optimised_random_forest.joblib")
    print("- results/figures/confusion_matrix_baseline.png")
    print("- results/figures/confusion_matrix_optimised.png")
    print("- results/figures/roc_curve_comparison.png")


if __name__ == "__main__":
    main()