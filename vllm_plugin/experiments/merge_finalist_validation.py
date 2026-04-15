#!/usr/bin/env python3
"""
Merge interrupted finalist-validation shards into a single consolidated artifact set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from vllm_plugin.experiments.run_finalist_validation import (
        _build_root_review_bundle,
        _build_variant_score_rows,
        _write_csv,
        _write_md,
    )
except ModuleNotFoundError:
    from run_finalist_validation import (
        _build_root_review_bundle,
        _build_variant_score_rows,
        _write_csv,
        _write_md,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


def _read_runs(runs_jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_jsonl_path.exists():
        return rows
    with runs_jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_variant_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Invalid --variant-artifact value: {value}")
    variant, artifact = value.split("=", 1)
    return variant.strip(), Path(artifact).resolve()


def _row_generated_at(row: dict[str, Any]) -> str:
    return str(row.get("generated_at") or "")


def _sort_rows(rows: list[dict[str, Any]], file_order: dict[str, int]) -> list[dict[str, Any]]:
    scenario_order = {"gold_full": 0}
    return sorted(
        rows,
        key=lambda row: (
            file_order.get(str(row.get("file_id")), 999999),
            scenario_order.get(str(row.get("scenario")), 1),
            str(row.get("scenario")),
        ),
    )


def _copy_transcript_files(
    *,
    src_artifact_dir: Path,
    dst_artifact_dir: Path,
    file_id: str,
    scenario: str,
) -> None:
    file_slug = _slugify(file_id)
    scenario_slug = _slugify(scenario)
    src_dir = src_artifact_dir / "transcripts" / file_slug
    dst_dir = dst_artifact_dir / "transcripts" / file_slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("json", "txt"):
        src_path = src_dir / f"{scenario_slug}.{suffix}"
        if src_path.exists():
            shutil.copy2(src_path, dst_dir / src_path.name)


def _merge_variant_rows(
    variant: str,
    artifact_dirs: list[Path],
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    source_records: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        rows = _read_runs(artifact_dir / "runs.jsonl")
        source_records.append(
            {
                "artifact_dir": str(artifact_dir),
                "row_count": len(rows),
            }
        )
        for row in rows:
            key = (str(row.get("file_id")), str(row.get("scenario")))
            if key not in chosen or _row_generated_at(row) >= _row_generated_at(chosen[key][0]):
                chosen[key] = (row, artifact_dir)

    merged_rows = [row for row, _ in chosen.values()]
    merged_rows.sort(key=lambda row: (_row_generated_at(row), str(row.get("file_id")), str(row.get("scenario"))))
    return artifact_dirs[0], merged_rows, source_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge finalist-validation shards into a consolidated result.")
    parser.add_argument(
        "--run-root",
        required=True,
        help="Root directory of the finalist validation run.",
    )
    parser.add_argument(
        "--variant-artifact",
        action="append",
        default=[],
        help="Variant to artifact mapping in the form variant=/abs/path/to/artifact_dir. Repeat per shard.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    manifest_path = run_root / "manifest_used.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run-root manifest: {manifest_path}")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_entries = manifest_payload.get("files", [])
    file_order = {
        str(entry.get("id") or entry.get("file_id")): index
        for index, entry in enumerate(file_entries)
        if isinstance(entry, dict)
    }
    scenarios = sorted(
        {
            f"chunk_{value}m_overlap_context"
            for value in manifest_payload.get("chunk_minutes", [30])
        }
    )

    explicit: dict[str, list[Path]] = {}
    for raw in args.variant_artifact:
        variant, artifact_dir = _parse_variant_artifact(raw)
        explicit.setdefault(variant, []).append(artifact_dir)

    if not explicit:
        raise ValueError("At least one --variant-artifact mapping is required.")

    merged_root = run_root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)

    variant_runs: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    merge_report: dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "run_root": str(run_root),
        "variants": {},
    }

    for variant, artifact_dirs in explicit.items():
        seed_artifact_dir, merged_rows, source_records = _merge_variant_rows(variant, artifact_dirs)
        merged_variant_dir = merged_root / variant
        merged_variant_dir.mkdir(parents=True, exist_ok=True)

        selected_sources: dict[tuple[str, str], Path] = {}
        for artifact_dir in artifact_dirs:
            for row in _read_runs(artifact_dir / "runs.jsonl"):
                key = (str(row.get("file_id")), str(row.get("scenario")))
                if key not in selected_sources:
                    selected_sources[key] = artifact_dir
                else:
                    prior_rows = _read_runs(selected_sources[key] / "runs.jsonl")
                    prior_match = next(
                        (
                            candidate
                            for candidate in prior_rows
                            if str(candidate.get("file_id")) == key[0] and str(candidate.get("scenario")) == key[1]
                        ),
                        None,
                    )
                    if prior_match is None or _row_generated_at(row) >= _row_generated_at(prior_match):
                        selected_sources[key] = artifact_dir

        ordered_rows = _sort_rows(merged_rows, file_order)
        _write_csv(merged_variant_dir / "summary.csv", ordered_rows)
        with (merged_variant_dir / "runs.jsonl").open("w", encoding="utf-8") as fp:
            for row in ordered_rows:
                fp.write(json.dumps(row, ensure_ascii=True) + "\n")

        manifest_out = {
            "generated_at": _utc_now_iso(),
            "variant": variant,
            "source_artifacts": [str(path) for path in artifact_dirs],
            "seed_artifact_dir": str(seed_artifact_dir),
        }
        (merged_variant_dir / "manifest_used.json").write_text(
            json.dumps(manifest_out, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        for row in ordered_rows:
            file_id = str(row.get("file_id"))
            scenario = str(row.get("scenario"))
            src_artifact_dir = selected_sources[(file_id, scenario)]
            _copy_transcript_files(
                src_artifact_dir=src_artifact_dir,
                dst_artifact_dir=merged_variant_dir,
                file_id=file_id,
                scenario=scenario,
            )

        variant_runs[variant] = (merged_variant_dir, ordered_rows)
        merge_report["variants"][variant] = {
            "sources": source_records,
            "merged_artifact_dir": str(merged_variant_dir),
            "selected_rows": len(ordered_rows),
        }

    scoreboard_rows = _build_variant_score_rows(variant_runs, scenarios=scenarios)
    scoreboard_csv = run_root / "variant_scoreboard.csv"
    scoreboard_md = run_root / "variant_scoreboard.md"
    _write_csv(scoreboard_csv, scoreboard_rows)
    _write_md(scoreboard_md, "Finalist Long-Call Metric Scoreboard", scoreboard_rows)
    _build_root_review_bundle(run_root, variant_runs, scenarios=scenarios)

    merge_report["paths"] = {
        "scoreboard_csv": str(scoreboard_csv),
        "scoreboard_md": str(scoreboard_md),
        "expert_review_template_json": str(run_root / "expert_review" / "review_template.json"),
    }
    (run_root / "merge_report.json").write_text(
        json.dumps(merge_report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[INFO] Merge complete.")
    print(f"[INFO] Run root: {run_root}")
    print(f"[INFO] Scoreboard: {scoreboard_csv}")


if __name__ == "__main__":
    main()
