QQQ-Relative Random Forest Portfolio Allocation
Project Overview

This project implements an end-to-end machine-learning pipeline for QQQ-relative stock selection and portfolio backtesting. The aim is to evaluate whether a Random Forest classifier can rank selected QQQ equities according to their probability of outperforming QQQ over the following month, and whether those rankings can be converted into portfolio allocation decisions that improve on benchmark strategies.

The project is structured as a supervised binary classification problem. Each observation represents one stock at one monthly decision date. The target variable equals 1 if the stock outperforms QQQ over the following month, and 0 otherwise.

The Random Forest model produces predicted probabilities of next-month QQQ-relative outperformance. These predicted probabilities are then used as stock-ranking scores. The highest-ranked stocks are selected into Top-K portfolios, which are evaluated against QQQ buy-and-hold and an equal-weight universe benchmark.

The implementation focuses on reproducibility, time-aware evaluation, portfolio-level testing, benchmark comparison, transaction-cost adjustment, robustness testing and feature-group ablation.

Final Implementation Scope

The final codebase includes the following components:

Historical price and volume data collection
Monthly feature engineering
QQQ-relative target construction
Baseline Random Forest classifier
Optimised Random Forest classifier
Time-aware validation
Classification evaluation
Probability-based stock ranking
Top-K portfolio backtesting
QQQ buy-and-hold benchmark comparison
Equal-weight universe benchmark comparison
Transaction-cost adjustment
Transaction-cost sensitivity analysis
Top-K robustness testing
Feature-group ablation testing
Final output audit and result summary

SHAP is not implemented in the final code version. The final implementation instead uses Random Forest feature importance and feature-group ablation testing to support interpretation of the model and its feature groups.

Project Structure

qqq_rf_shap_dissertation
│
├── README.md
├── config.yaml
├── requirements.txt
│
├── data
│ ├── raw
│ │ └── prices.csv
│ └── processed
│ └── model_panel.csv
│
├── results
│ ├── tables
│ ├── figures
│ ├── models
│ └── logs
│
├── scripts
│ ├── 01_download_prices.py
│ ├── 02_build_features.py
│ ├── 03_train_random_forest.py
│ ├── 04_run_backtest.py
│ ├── 05_run_robustness_ablation.py
│ └── 06_generate_final_outputs.py
│
└── src

Python Requirements

The project uses Python and the following main packages:

pandas
numpy
scikit-learn
matplotlib
yfinance
scipy
statsmodels
pyyaml
joblib

To install the required packages, run the following command from the main project folder:

python -m pip install -r requirements.txt

All scripts should be run using python, not py.

How to Run the Project

All commands should be run from the main project folder:

C:\Users\jason\Downloads\qqq_rf_shap_dissertation_backup (1)\qqq_rf_shap_dissertation

Step 1: Download price data

Run:

python scripts/01_download_prices.py

This script downloads historical price and volume data for QQQ and the selected QQQ equities.

Main output:

data/raw/prices.csv

Step 2: Build features and target variable

Run:

python scripts/02_build_features.py

This script converts the raw daily price data into a monthly stock-level model panel. It calculates market, momentum, volume, risk and QQQ-relative features. It also creates the binary target variable outperform_qqq_next_1m.

Main outputs:

data/processed/model_panel.csv
results/tables/dataset_summary.csv
results/tables/class_balance.csv
results/tables/feature_list.csv

Step 3: Train Random Forest models

Run:

python scripts/03_train_random_forest.py

This script trains both a baseline Random Forest classifier and an optimised Random Forest classifier. The optimised model is selected using hyperparameter search and time-aware validation.

Main outputs:

results/tables/model_metrics.csv
results/tables/rf_predictions.csv
results/tables/selected_hyperparameters.csv
results/tables/optimisation_search_results.csv
results/tables/optimised_rf_feature_importance.csv
results/models/baseline_random_forest.joblib
results/models/optimised_random_forest.joblib
results/figures/confusion_matrix_baseline.png
results/figures/confusion_matrix_optimised.png
results/figures/roc_curve_comparison.png

Step 4: Run portfolio backtest

Run:

python scripts/04_run_backtest.py

This script converts Random Forest predicted probabilities into Top-K portfolios. It compares the baseline Random Forest portfolio and the optimised Random Forest portfolio against QQQ buy-and-hold and an equal-weight universe benchmark. It also applies transaction-cost adjustments.

Main outputs:

results/tables/portfolio_monthly_returns.csv
results/tables/portfolio_holdings.csv
results/tables/benchmark_monthly_returns.csv
results/tables/portfolio_performance.csv
results/tables/benchmark_comparison.csv
results/tables/portfolio_comparison_monthly_returns.csv
results/tables/transaction_cost_sensitivity.csv
results/figures/cumulative_returns_comparison.png
results/figures/drawdown_comparison.png

Step 5: Run robustness and ablation tests

Run:

python scripts/05_run_robustness_ablation.py

This script tests whether the results are sensitive to portfolio size and feature-group inclusion. It evaluates Top-K robustness and feature-group ablation models.

Main outputs:

results/tables/topk_robustness_results.csv
results/tables/ablation_classification_results.csv
results/tables/ablation_portfolio_results.csv
results/tables/ablation_predictions.csv
results/figures/topk_robustness_total_return.png
results/figures/ablation_total_return.png

Step 6: Generate final implementation audit

Run:

python scripts/06_generate_final_outputs.py

This script checks whether the required result files exist and produces a final summary of the key implementation outputs.

Main outputs:

results/tables/final_implementation_audit.csv
results/tables/final_key_results_summary.csv
results/logs/final_results_summary.txt

Dataset Summary

The final model panel contains the following dataset characteristics:

Benchmark: QQQ
Stock-month observations: 2,520
Unique stocks: 20
Sample start date: 2016-01-31
Sample end date: 2026-06-30
Predictive features: 15
Target variable: outperform_qqq_next_1m
Positive class count: 1,292
Negative class count: 1,228
Positive class percentage: 51.27%
Negative class percentage: 48.73%

The class balance is relatively even, meaning that accuracy is not heavily distorted by a strongly imbalanced target variable. However, accuracy is still interpreted alongside precision, recall, F1-score and ROC-AUC.

Feature Groups

The final model uses market-based features constructed from historical adjusted prices, volume and QQQ benchmark returns.

The feature groups include:

Momentum features:
ret_1m, ret_3m, ret_6m, ret_12m

Volume feature:
volume_growth_3m

Risk features:
volatility_1m, volatility_3m, beta_3m, corr_qqq_3m, tracking_error_3m, max_drawdown_3m

QQQ-relative features:
relative_return_1m, relative_return_3m, relative_return_6m, relative_return_12m

The target variable is:

outperform_qqq_next_1m

Financial ratios are not included in the final feature set because reliable point-in-time accounting data was not used in the final implementation. This reduces the risk of look-ahead bias from restated or retrospectively available financial-ratio data.

Model Design

The implemented model is a Random Forest classifier. Random Forest is used because it is suitable for structured tabular data, can capture non-linear relationships, can model feature interactions, and produces probability estimates that can be used as ranking scores.

The project trains two Random Forest models:

Baseline Random Forest
Optimised Random Forest

The baseline model provides a simple reference model. The optimised model uses selected hyperparameters from the search process and is used as the main model for portfolio construction.

The prediction target is not absolute stock return. Instead, the model predicts whether each stock will outperform QQQ over the following month.

The model output is therefore:

P(stock outperforms QQQ next month | current features)

This predicted probability is used as the ranking score for portfolio construction.

Validation Design

The implementation uses time-aware validation rather than a random train-test split. This is important because financial data has a temporal structure. Randomly mixing earlier and later observations could create an unrealistic evaluation design.

The model is trained on historical observations and evaluated on later observations. This design helps reduce look-ahead bias and makes the evaluation closer to how the model would be used in practice.

The final outputs include model metrics, predictions, selected hyperparameters, confusion matrices and a ROC curve comparison.

Portfolio Construction

The portfolio construction stage converts predicted probabilities into Top-K portfolios.

At each monthly decision date:

The model assigns a predicted probability to each stock.
Stocks are ranked from highest to lowest predicted probability.
The Top-K stocks are selected.
The selected stocks are equal-weighted.
The portfolio return is calculated using the next-month realised stock returns.
Transaction costs are applied based on portfolio turnover.

The main strategy is the optimised Random Forest Top-K portfolio.

The model-based strategies are compared against:

QQQ buy-and-hold
Equal-weight universe benchmark
Baseline Random Forest Top-K portfolio
Transaction Costs

Transaction costs are included because a strategy that performs well before costs may not remain attractive after trading frictions.

The implementation calculates portfolio turnover and subtracts transaction costs from gross returns. The results include both gross and net performance, together with transaction-cost sensitivity analysis.

The transaction-cost sensitivity analysis tests how performance changes under different basis-point assumptions.

Main Classification Results

The optimised Random Forest produced the strongest classification result among the implemented models. However, the classification performance was modest.

Optimised Random Forest classification results:

Accuracy: 51.46%
Precision: 47.64%
Recall: 59.55%
F1-score: 52.93%
ROC-AUC: 52.11%

These results suggest that the model has only limited ability to classify individual stock-month observations correctly. The ROC-AUC is slightly above 0.50, meaning that the model’s ranking ability is only modest at the individual-observation level.

However, the classification results should not be interpreted in isolation. The dissertation also evaluates whether the model’s probability rankings can support portfolio allocation.

Main Portfolio Results

The optimised Random Forest Top-K portfolio achieved the strongest portfolio performance in the main portfolio-performance table.

Optimised Random Forest Top-K results:

Total return: 65.87%
Annualised return: 28.79%
Annualised volatility: 20.83%
Sharpe ratio: 1.3821
Sortino ratio: 2.5762
Maximum drawdown: -11.51%
Calmar ratio: 2.5021
Average monthly turnover: 83.33%

The optimised Random Forest Top-K portfolio outperformed the baseline Random Forest Top-K portfolio, QQQ buy-and-hold and the equal-weight universe benchmark in total return.

Performance differences:

Optimised RF minus baseline RF total return: +11.99 percentage points
Optimised RF minus QQQ buy-and-hold total return: +16.56 percentage points
Optimised RF minus equal-weight universe total return: +12.25 percentage points

These results suggest that although the model’s classification performance was modest, its predicted probabilities still had value as ranking signals for portfolio construction.

Robustness Results

The Top-K robustness test evaluates whether the portfolio results depend heavily on the chosen number of holdings.

The best Top-K robustness configuration was:

Optimised Random Forest Top-5
Total return: 78.62%

This indicates that the highest-ranked subset of stocks produced the strongest total return in the Top-K robustness test. However, this result should be interpreted carefully because smaller portfolios may be more concentrated and may carry higher idiosyncratic risk.

Ablation Results

Feature-group ablation tests evaluate whether removing particular feature groups affects classification and portfolio performance.

The best ablation model by portfolio total return was:

without_qqq_relative
Total return: 74.40%

The best ablation model by ROC-AUC was also:

without_qqq_relative
ROC-AUC: 52.37%

This suggests that the QQQ-relative feature group may not have improved model performance in the final implementation. One possible explanation is that QQQ-relative features may have introduced redundancy or noise, especially because the target variable was already defined relative to QQQ.

The ablation results should be interpreted as diagnostic evidence rather than as proof that QQQ-relative features are generally useless. The finding applies to this dataset, feature set, model design and sample period.

Interpretation of Results

The main empirical finding is that the Random Forest models produced modest classification metrics but stronger portfolio-level results.

This distinction is important. In a portfolio setting, a model does not need to classify every stock-month observation correctly to be useful. It may still add value if the highest-ranked stocks perform better than the benchmark or better than a naive allocation strategy.

The optimised Random Forest Top-K portfolio produced stronger total return than QQQ buy-and-hold, the equal-weight universe benchmark and the baseline Random Forest portfolio. This suggests that the probability rankings had some economic value when translated into portfolio allocation decisions.

However, the modest ROC-AUC means that the results should be interpreted cautiously. The model should not be presented as a highly accurate predictor of stock outperformance. A more appropriate interpretation is that the framework provides a structured, reproducible method for testing whether machine-learning probability rankings can support benchmark-relative allocation.

Limitations

The implementation has several limitations.

First, the stock universe contains selected QQQ equities rather than a full point-in-time historical Nasdaq-100 constituent universe. This means the analysis may be affected by survivorship bias.

Second, the model uses market-based features rather than fully point-in-time accounting ratios. This improves timing validity but excludes potentially useful fundamental information.

Third, the classification results are modest. The ROC-AUC is only slightly above 0.50, which means the model has limited individual-observation prediction strength.

Fourth, transaction costs are modelled using simplified basis-point assumptions. Real-world implementation costs may vary depending on liquidity, spread, market impact, trade size and execution quality.

Fifth, the portfolio backtest is historical. It does not prove that the strategy will perform well in the future.

Sixth, SHAP explainability is not included in the final code implementation. Therefore, the report should not claim that SHAP analysis was implemented. Interpretation is instead supported through Random Forest feature importance and feature-group ablation.

Reproducibility Notes

The implementation is designed so that each stage can be run independently and inspected through saved outputs.

Raw data is saved separately from processed data. Processed model inputs are saved separately from results. Model outputs, figures, tables and logs are stored in the results folder.

The final audit script checks that all required outputs exist. In the final run, the audit checked 25 required files and found all 25 files, with 0 missing files.

This confirms that the main implementation pipeline generated the expected output set.

Final Technical Claim

This project provides a reproducible, time-aware Random Forest framework for QQQ-relative stock ranking and portfolio allocation.

The implementation connects:

historical data collection
feature engineering
supervised learning
hyperparameter optimisation
probability-based stock ranking
Top-K portfolio construction
benchmark comparison
transaction-cost adjustment
robustness testing
feature-group ablation
final output auditing

The project should be interpreted as an academic machine-learning and portfolio-evaluation pipeline. It is not financial advice and does not claim to provide a guaranteed investment strategy.