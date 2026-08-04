# Prospective evaluation v1

This directory establishes a genuinely prospective evaluation beginning after the protocol freeze on 4 August 2026. Historical observations already encountered during development remain retrospective and must not be relabelled as prospective evidence.

## Evidence boundary

The first eligible decision date is 31 August 2026. Its one-month outcome becomes observable at the end of September 2026. Results before six completed decision months are labelled `prospective_monitoring_only`; results after six completed months are `preliminary_prospective_evidence`; results after twelve completed months are `primary_prospective_evidence`.

The prospective window uses the frozen feature schema, fitted models, Top-10 equal-weight portfolio rule, monthly rebalancing, QQQ benchmark and 10-basis-point turnover cost. No feature, model, hyperparameter, selection, weighting or cost change is permitted within this protocol.

## Step 1: create the lock manifest once

Run from the repository root:

```bash
python scripts/20_lock_prospective_protocol.py
```

This creates:

```text
prospective/prospective_lock_manifest.json
```

The command refuses to overwrite an existing lock. It records SHA-256 hashes for the protocol, configuration, code and frozen model artefacts available at lock time.

Commit and, where possible, tag the manifest immediately. The Git commit time and file hashes establish the pre-outcome protocol state.

## Step 2: generate one month of predictions

At each eligible month-end, use only information available by that decision date. Produce a CSV with one row per ticker and model containing:

```text
date,ticker,model,score
```

Do not include the target, next-month stock return or next-month QQQ return. Predictions must be recorded before the corresponding outcome is known.

## Step 3: append predictions without overwrite

```bash
python scripts/21_append_prospective_predictions.py path/to/predictions_YYYY_MM.csv
```

The script:

- verifies every locked file hash;
- rejects realised-outcome columns;
- accepts exactly one decision month;
- rejects dates before 31 August 2026;
- rejects duplicate date-ticker-model keys;
- refuses to overwrite a previously recorded month;
- appends the rows to a prospective ledger; and
- writes an immutable monthly archive.

Outputs:

```text
prospective/results/prospective_predictions.csv
prospective/results/predictions_YYYY_MM.csv
```

## Step 4: reveal outcomes later

Only after the full next-month outcome is available should realised stock and benchmark returns be joined to the archived prediction file. Outcome evaluation should be implemented as a separate append-only step so that predictions cannot be recomputed after observing returns.

## Governance rules

Any model retraining, new feature, new universe, changed transaction cost, altered Top-K rule or changed weighting method creates a new protocol version. It must not overwrite or be pooled silently with v1.

Negative, null and adverse outcomes are retained. A failure to outperform QQQ or equal weighting is valid prospective evidence and must not trigger retrospective model replacement within the locked window.

## Dissertation wording

Until prospective outcomes exist, the report may state that a prospective protocol has been specified and technically locked. It must not claim prospective validation or any prospective performance result. After sufficient observations accumulate, the report should disclose the freeze date, first eligible decision date, number of completed months, any missing decisions and the exact evidence label defined above.
