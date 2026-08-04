from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_PREDICTION_COLUMNS = [
    "date",
    "ticker",
    "model",
    "score",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_lock(lock: dict) -> None:
    for entry in lock.get("files", []):
        path = Path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Locked file is missing: {path}")
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise RuntimeError(
                f"Locked file changed: {path}. Expected {entry['sha256']}, "
                f"observed {observed}."
            )


def normalise_month_end(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="raise")
    return dates.dt.to_period("M").dt.to_timestamp("M")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and append one frozen prospective prediction month. "
            "Existing rows and files are never overwritten."
        )
    )
    parser.add_argument("prediction_file", help="CSV containing one decision month")
    parser.add_argument(
        "--protocol",
        default="prospective/protocol_v1.yaml",
    )
    parser.add_argument(
        "--lock",
        default="prospective/prospective_lock_manifest.json",
    )
    parser.add_argument(
        "--ledger",
        default="prospective/results/prospective_predictions.csv",
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    lock_path = Path(args.lock)
    prediction_path = Path(args.prediction_file)
    ledger_path = Path(args.ledger)

    protocol = load_yaml(protocol_path)
    lock = load_json(lock_path)
    verify_lock(lock)

    frame = pd.read_csv(prediction_path)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame]
    if missing:
        raise ValueError("Prediction file is missing columns: " + ", ".join(missing))

    forbidden = {
        "future_stock_return",
        "future_qqq_return",
        "outperform_qqq_next_1m",
    }
    leaked = sorted(forbidden.intersection(frame.columns))
    if leaked:
        raise ValueError(
            "Prediction file contains realised-outcome columns and cannot be "
            "recorded prospectively: " + ", ".join(leaked)
        )

    frame = frame.copy()
    frame["date"] = normalise_month_end(frame["date"])
    decision_dates = frame["date"].drop_duplicates().sort_values()
    if len(decision_dates) != 1:
        raise ValueError("Exactly one decision month must be supplied per append.")

    decision_date = decision_dates.iloc[0]
    first_eligible = pd.Timestamp(
        protocol["scope"]["first_eligible_decision_month_end"]
    )
    if decision_date < first_eligible:
        raise ValueError(
            f"Decision date {decision_date.date()} predates the prospective "
            f"window beginning {first_eligible.date()}."
        )

    if frame.duplicated(["date", "ticker", "model"]).any():
        raise ValueError("Duplicate date-ticker-model prediction keys detected.")
    if frame["score"].isna().any():
        raise ValueError("Missing prediction scores detected.")

    frame["protocol_name"] = protocol["protocol"]["name"]
    frame["protocol_sha256"] = sha256_file(protocol_path)
    frame["source_file_sha256"] = sha256_file(prediction_path)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        existing = pd.read_csv(ledger_path)
        existing_dates = set(pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d"))
        date_text = decision_date.strftime("%Y-%m-%d")
        if date_text in existing_dates:
            raise RuntimeError(
                f"Decision month {date_text} already exists in the ledger; "
                "overwrite is prohibited."
            )
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame

    temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(ledger_path)

    month_copy = ledger_path.parent / f"predictions_{decision_date:%Y_%m}.csv"
    if month_copy.exists():
        raise FileExistsError(f"Monthly archive already exists: {month_copy}")
    frame.to_csv(month_copy, index=False)

    print(f"Appended prospective month: {decision_date.date()}")
    print(f"Rows appended: {len(frame)}")
    print(f"Ledger: {ledger_path}")
    print(f"Immutable monthly archive: {month_copy}")


if __name__ == "__main__":
    main()
