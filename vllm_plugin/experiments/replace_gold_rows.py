#!/usr/bin/env python3
"""Replace failed gold_full rows in a runs.jsonl file with successful rerun rows."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_source_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("Source rows must be JSON list or JSONL file.")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace failed gold_full rows in target runs.jsonl."
    )
    parser.add_argument(
        "--target-runs",
        required=True,
        help="Target runs.jsonl path to patch in place.",
    )
    parser.add_argument(
        "--source-rows",
        required=True,
        help="Source rerun rows (.json list or .jsonl).",
    )
    parser.add_argument(
        "--file-ids",
        nargs="+",
        required=True,
        help="File IDs to replace (gold_full scenario).",
    )
    parser.add_argument(
        "--scenario",
        default="gold_full",
        help="Scenario name to replace (default: gold_full).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    target_path = Path(args.target_runs).resolve()
    source_path = Path(args.source_rows).resolve()
    file_ids = set(args.file_ids)
    scenario = str(args.scenario)

    target_rows = _read_jsonl(target_path)
    source_rows = _read_source_rows(source_path)

    replacement_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_rows:
        key = (str(row.get("file_id")), str(row.get("scenario")))
        if key[0] not in file_ids or key[1] != scenario:
            continue
        if bool(row.get("success")):
            replacement_map[key] = row

    missing = [
        fid for fid in sorted(file_ids) if (fid, scenario) not in replacement_map
    ]
    if missing:
        raise RuntimeError(
            f"Missing successful replacement rows for: {', '.join(missing)}"
        )

    replaced = 0
    for i, row in enumerate(target_rows):
        key = (str(row.get("file_id")), str(row.get("scenario")))
        if key in replacement_map and bool(row.get("success")) is False:
            target_rows[i] = replacement_map[key]
            replaced += 1

    if replaced == 0:
        raise RuntimeError("No failed rows were replaced in target runs.jsonl.")

    backup_path = target_path.with_suffix(target_path.suffix + f".bak.{_utc_ts()}")
    shutil.copy2(target_path, backup_path)

    with target_path.open("w", encoding="utf-8") as fp:
        for row in target_rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[INFO] Backup: {backup_path}")
    print(f"[INFO] Replaced rows: {replaced}")
    print(f"[INFO] Patched target: {target_path}")


if __name__ == "__main__":
    main()
