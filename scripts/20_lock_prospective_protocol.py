from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_INPUTS = (
    "prospective/protocol_v1.yaml",
    "config.yaml",
    "requirements.txt",
    "scripts/02_build_features.py",
    "scripts/03_train_random_forest.py",
    "scripts/04_run_backtest.py",
    "results/models/baseline_random_forest.joblib",
    "results/models/optimised_random_forest.joblib",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(paths: Iterable[Path]) -> dict:
    entries = []
    missing = []

    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        if not path.is_file():
            raise ValueError(f"Expected a file, found something else: {path}")
        entries.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED",
        "files": entries,
        "missing_optional_or_unavailable_files": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an append-only manifest for the prospective evaluation. "
            "The command refuses to overwrite an existing lock."
        )
    )
    parser.add_argument(
        "--output",
        default="prospective/prospective_lock_manifest.json",
        help="Manifest path. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=list(DEFAULT_INPUTS),
        help="Files to fingerprint.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Prospective lock already exists at {output}. "
            "Delete or overwrite is intentionally prohibited."
        )

    input_paths = [Path(item) for item in args.inputs]
    manifest = build_manifest(input_paths)

    required = {
        "prospective/protocol_v1.yaml",
        "config.yaml",
        "requirements.txt",
    }
    present = {entry["path"] for entry in manifest["files"]}
    missing_required = sorted(required - present)
    if missing_required:
        raise FileNotFoundError(
            "Cannot create lock; required files are missing: "
            + ", ".join(missing_required)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Created prospective lock: {output}")
    print(f"Fingerprinted files: {len(manifest['files'])}")
    if manifest["missing_optional_or_unavailable_files"]:
        print("Unavailable files recorded in manifest:")
        for path in manifest["missing_optional_or_unavailable_files"]:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
