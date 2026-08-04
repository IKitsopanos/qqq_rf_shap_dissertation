"""Point-in-time index membership and security-master utilities.

The module deliberately separates permanent security identity from ticker symbols.
It does not download or infer historical membership. Instead, it validates a
provenance-bearing security master and expands membership intervals into monthly
eligibility records that can be joined to model panels without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SECURITY_MASTER_COLUMNS = {
    "security_id",
    "ticker",
    "company_name",
    "index_name",
    "entry_date",
    "exit_date",
    "ticker_start_date",
    "ticker_end_date",
    "delisting_date",
    "delisting_return",
    "event_type",
    "source_name",
    "source_url",
    "source_retrieved_at",
    "source_quality",
}

DATE_COLUMNS = [
    "entry_date",
    "exit_date",
    "ticker_start_date",
    "ticker_end_date",
    "delisting_date",
    "source_retrieved_at",
]

ALLOWED_SOURCE_QUALITY = {"official", "licensed", "archival", "secondary", "unverified"}


@dataclass(frozen=True)
class UniverseAudit:
    index_name: str
    start_month: pd.Timestamp
    end_month: pd.Timestamp
    membership_rows: int
    unique_securities: int
    unique_tickers: int
    months: int
    missing_source_urls: int
    unverified_rows: int
    missing_delisting_returns: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


def _normalise_month(value: pd.Timestamp | str) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp("M")


def load_security_master(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_security_master(frame)


def validate_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a provenance-bearing security master.

    One row represents one ticker spell over one index-membership interval. A
    permanent security can therefore appear on several rows when its ticker or
    index membership changes.
    """
    missing = sorted(SECURITY_MASTER_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Security master is missing required columns: {missing}")

    out = frame.copy()
    for column in DATE_COLUMNS:
        out[column] = pd.to_datetime(out[column], errors="coerce", utc=False)

    out["security_id"] = out["security_id"].astype(str).str.strip()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["index_name"] = out["index_name"].astype(str).str.upper().str.strip()
    out["source_quality"] = out["source_quality"].astype(str).str.lower().str.strip()
    out["event_type"] = out["event_type"].fillna("membership").astype(str).str.lower().str.strip()
    out["delisting_return"] = pd.to_numeric(out["delisting_return"], errors="coerce")

    if (out["security_id"] == "").any() or (out["ticker"] == "").any():
        raise ValueError("security_id and ticker must be non-empty")
    if out["entry_date"].isna().any():
        raise ValueError("entry_date is required for every membership row")
    if (~out["source_quality"].isin(ALLOWED_SOURCE_QUALITY)).any():
        invalid = sorted(out.loc[~out["source_quality"].isin(ALLOWED_SOURCE_QUALITY), "source_quality"].unique())
        raise ValueError(f"Invalid source_quality values: {invalid}")

    effective_exit = out["exit_date"].fillna(pd.Timestamp.max.normalize())
    if (effective_exit < out["entry_date"]).any():
        raise ValueError("exit_date cannot precede entry_date")

    ticker_start = out["ticker_start_date"].fillna(out["entry_date"])
    ticker_end = out["ticker_end_date"].fillna(out["exit_date"])
    effective_ticker_end = ticker_end.fillna(pd.Timestamp.max.normalize())
    if (effective_ticker_end < ticker_start).any():
        raise ValueError("ticker_end_date cannot precede ticker_start_date")

    duplicate_key = ["security_id", "ticker", "index_name", "entry_date", "exit_date"]
    if out.duplicated(duplicate_key, keep=False).any():
        dupes = out.loc[out.duplicated(duplicate_key, keep=False), duplicate_key]
        raise ValueError(f"Duplicate membership intervals detected:\n{dupes.to_string(index=False)}")

    out["ticker_start_date"] = ticker_start
    out["ticker_end_date"] = ticker_end
    return out.sort_values(["index_name", "security_id", "entry_date", "ticker_start_date"]).reset_index(drop=True)


def expand_monthly_membership(
    security_master: pd.DataFrame,
    index_name: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Expand validated membership intervals to month-end eligibility rows.

    Eligibility is evaluated at each calendar month-end. A row is included only
    where both the index-membership interval and ticker-validity interval cover
    that month-end. No attempt is made to backfill a missing history.
    """
    master = validate_security_master(security_master)
    index_name = index_name.upper().strip()
    start_month = _normalise_month(start_date)
    end_month = _normalise_month(end_date)
    if end_month < start_month:
        raise ValueError("end_date must be on or after start_date")

    subset = master.loc[master["index_name"] == index_name].copy()
    if subset.empty:
        raise ValueError(f"No security-master rows found for index {index_name}")

    records: list[dict] = []
    for row in subset.itertuples(index=False):
        interval_start = max(_normalise_month(row.entry_date), start_month)
        membership_end = end_month if pd.isna(row.exit_date) else _normalise_month(row.exit_date)
        ticker_start = _normalise_month(row.ticker_start_date)
        ticker_end = end_month if pd.isna(row.ticker_end_date) else _normalise_month(row.ticker_end_date)
        interval_start = max(interval_start, ticker_start)
        interval_end = min(membership_end, ticker_end, end_month)
        if interval_end < interval_start:
            continue

        for month in pd.date_range(interval_start, interval_end, freq="ME"):
            records.append(
                {
                    "date": month,
                    "security_id": row.security_id,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "index_name": row.index_name,
                    "entry_date": row.entry_date,
                    "exit_date": row.exit_date,
                    "delisting_date": row.delisting_date,
                    "delisting_return": row.delisting_return,
                    "event_type": row.event_type,
                    "source_name": row.source_name,
                    "source_url": row.source_url,
                    "source_retrieved_at": row.source_retrieved_at,
                    "source_quality": row.source_quality,
                }
            )

    membership = pd.DataFrame.from_records(records)
    if membership.empty:
        raise ValueError("Membership expansion produced no eligible rows")

    key = ["date", "security_id", "index_name"]
    conflicts = membership.duplicated(key, keep=False)
    if conflicts.any():
        rows = membership.loc[conflicts, key + ["ticker"]]
        raise ValueError(
            "Overlapping intervals create multiple ticker records for the same security-month:\n"
            + rows.to_string(index=False)
        )

    return membership.sort_values(["date", "security_id"]).reset_index(drop=True)


def filter_panel_to_point_in_time_universe(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    panel_date_col: str = "date",
    panel_ticker_col: str = "ticker",
    strict: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join a model panel to monthly historical membership.

    Returns the eligible panel and an audit table of panel rows that failed to
    match historical membership. In strict mode, duplicate panel keys or
    duplicate membership keys are rejected.
    """
    required_panel = {panel_date_col, panel_ticker_col}
    missing_panel = required_panel.difference(panel.columns)
    if missing_panel:
        raise ValueError(f"Panel is missing required columns: {sorted(missing_panel)}")

    panel_norm = panel.copy()
    panel_norm[panel_date_col] = pd.to_datetime(panel_norm[panel_date_col]).dt.to_period("M").dt.to_timestamp("M")
    panel_norm[panel_ticker_col] = panel_norm[panel_ticker_col].astype(str).str.upper().str.strip()

    membership_norm = membership.copy()
    membership_norm["date"] = pd.to_datetime(membership_norm["date"]).dt.to_period("M").dt.to_timestamp("M")
    membership_norm["ticker"] = membership_norm["ticker"].astype(str).str.upper().str.strip()

    if strict and panel_norm.duplicated([panel_date_col, panel_ticker_col]).any():
        raise ValueError("Panel contains duplicate date-ticker rows")
    if strict and membership_norm.duplicated(["date", "ticker"]).any():
        raise ValueError("Membership contains duplicate date-ticker rows")

    merged = panel_norm.merge(
        membership_norm,
        how="left",
        left_on=[panel_date_col, panel_ticker_col],
        right_on=["date", "ticker"],
        suffixes=("", "_membership"),
        indicator=True,
    )
    audit = merged.loc[merged["_merge"] != "both", [panel_date_col, panel_ticker_col, "_merge"]].copy()
    eligible = merged.loc[merged["_merge"] == "both"].drop(columns=["_merge", "date_membership", "ticker_membership"], errors="ignore")
    return eligible.reset_index(drop=True), audit.reset_index(drop=True)


def apply_delisting_returns(
    panel: pd.DataFrame,
    *,
    return_col: str = "future_stock_return",
    date_col: str = "date",
) -> pd.DataFrame:
    """Apply documented delisting returns on the corresponding outcome month.

    A delisting correction is used only when a delisting date and return are
    present. Existing returns are replaced because vendor prices commonly omit
    the terminal delisting payoff. The original value is retained for audit.
    """
    required = {return_col, date_col, "delisting_date", "delisting_return"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Cannot apply delisting returns; missing columns: {sorted(missing)}")

    out = panel.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.to_period("M").dt.to_timestamp("M")
    out["delisting_date"] = pd.to_datetime(out["delisting_date"], errors="coerce")
    delisting_month = out["delisting_date"].dt.to_period("M").dt.to_timestamp("M")
    mask = delisting_month.eq(out[date_col]) & out["delisting_return"].notna()
    out[f"{return_col}_before_delisting_adjustment"] = out[return_col]
    out["delisting_adjustment_applied"] = mask
    out.loc[mask, return_col] = out.loc[mask, "delisting_return"]
    return out


def audit_membership(
    membership: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> UniverseAudit:
    start_month = _normalise_month(start_date)
    end_month = _normalise_month(end_date)
    expected_months = len(pd.date_range(start_month, end_month, freq="ME"))
    delisted = membership["delisting_date"].notna()
    missing_delisting_returns = int((delisted & membership["delisting_return"].isna()).sum())
    return UniverseAudit(
        index_name=str(membership["index_name"].iloc[0]),
        start_month=start_month,
        end_month=end_month,
        membership_rows=len(membership),
        unique_securities=membership["security_id"].nunique(),
        unique_tickers=membership["ticker"].nunique(),
        months=expected_months,
        missing_source_urls=int(membership["source_url"].fillna("").eq("").sum()),
        unverified_rows=int(membership["source_quality"].eq("unverified").sum()),
        missing_delisting_returns=missing_delisting_returns,
    )


def write_membership_outputs(
    membership: pd.DataFrame,
    output_dir: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_dir / "monthly_membership.csv", index=False)
    audit_membership(membership, start_date, end_date).to_frame().to_csv(
        output_dir / "membership_audit.csv", index=False
    )
