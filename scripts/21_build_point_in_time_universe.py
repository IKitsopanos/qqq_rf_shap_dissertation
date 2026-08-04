#!/usr/bin/env python3
"""Build and audit monthly point-in-time index membership files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.point_in_time_universe import (  # noqa: E402
    expand_monthly_membership,
    load_security_master,
    write_membership_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand security-master membership intervals into monthly point-in-time eligibility."
    )
    parser.add_argument("--security-master", required=True, type=Path)
    parser.add_argument("--index", required=True, dest="index_name")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = load_security_master(args.security_master)
    membership = expand_monthly_membership(
        master,
        index_name=args.index_name,
        start_date=args.start,
        end_date=args.end,
    )
    write_membership_outputs(
        membership,
        output_dir=args.output_dir,
        start_date=args.start,
        end_date=args.end,
    )
    print(f"Index: {args.index_name.upper()}")
    print(f"Monthly membership rows: {len(membership):,}")
    print(f"Unique securities: {membership['security_id'].nunique():,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
