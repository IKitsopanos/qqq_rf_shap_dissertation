# Dissertation Upgrade Roadmap

This roadmap converts the ten proposed dissertation improvements into auditable engineering workstreams. Changes must preserve chronological validity, provenance, reproducibility and honest evidential status.

## Workstream 1 — Point-in-time universes and delisting treatment

Status: **in progress on `upgrade/point-in-time-universe`**

Deliverables:

- provenance-bearing security master keyed by permanent security identifier;
- monthly historical membership for Nasdaq-100 and S&P 100;
- ticker-history, entry, exit, merger and delisting fields;
- deterministic eligibility filter for model panels;
- documented delisting-return correction;
- survivor-selected versus point-in-time comparison;
- membership and provenance audit tables.

Acceptance criteria:

- no duplicate security-month memberships;
- no model row before index entry or after index exit;
- every membership interval has a named source and quality classification;
- missing delisting returns are counted and disclosed;
- synthetic interval, ticker-change and delisting tests pass.

## Workstream 2 — Portfolio-level inference and factor attribution

Planned deliverables:

- circular month-block bootstrap for portfolio paths;
- confidence intervals for active return, Sharpe, information ratio, drawdown, turnover and terminal wealth;
- predeclared primary comparison;
- Holm or SPA-family multiple-comparison control;
- Fama–French five-factor plus momentum attribution with Newey–West standard errors.

## Workstream 3 — Direct ranking and return-regression formulations

Planned deliverables:

- binary classification baseline retained;
- direct next-month relative-return regression;
- one learning-to-rank model with month as the query group;
- raw, cross-sectional rank and cross-sectional z-score preprocessing variants;
- common outer folds and portfolio rules across formulations.

## Workstream 4 — Grouped ablation and importance stability

Planned deliverables:

- leave-one-feature-group-out retraining inside each outer fold;
- out-of-sample permutation importance;
- fold and universe stability statistics;
- direct RQ5 synthesis table.

## Workstream 5 — Prospective protocol

Planned deliverables:

- frozen commit, environment and protocol;
- timestamped registration package;
- immutable future predictions;
- complete retention of successful, failed and all-cash decisions.

Historical isolation must not be described as prospective evaluation.

## Workstream 6 — Figure and presentation programme

Planned deliverables:

- wealth, active wealth, drawdown, monthly active return, reliability, IC, turnover and HHI figures;
- point-in-time versus survivor comparison figures;
- complete experimental-flow diagram;
- equation, numbering and table-order repairs.

## Workstream 7 — Related work, platforms and bibliography

Planned deliverables:

- quantitative related-work comparison table;
- short comparison with qlib, Zipline, backtrader, vectorbt, QuantConnect, MLflow and DVC;
- removal of unused references;
- addition of learning-to-rank and point-in-time data literature.

## Workstream 8 — Independent reproducibility

Planned deliverables:

- locked dependency environment;
- one-command reproduction target;
- continuous integration;
- leakage, portfolio accounting and optimiser tests;
- script-to-table provenance map;
- archived release and checksums.

## Workstream 9 — Execution and cost model

Planned deliverables:

- signal-to-trade delay;
- next-open or next-close execution;
- spread, slippage and market-impact decomposition;
- liquidity constraints and capacity analysis;
- break-even transaction costs.

## Workstream 10 — LSEP and governance hardening

Planned deliverables:

- named data and software licences;
- redistribution restrictions;
- ethics-evidence status;
- model-risk register;
- deployment, suspension and rollback responsibilities;
- compute-usage estimate.

## Sequencing

1. Point-in-time universe foundation.
2. Portfolio inference and attribution.
3. Ranking/regression extensions.
4. Ablation and importance stability.
5. Execution model.
6. Reproducibility package.
7. Figures, related work and report revision.
8. LSEP hardening.
9. Prospective evaluation continues after the frozen protocol date.
