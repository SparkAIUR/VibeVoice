#!/usr/bin/env python3
"""
Refresh a long-tail finalist run with chunked reference-gold transcripts.

This utility is intended for >60 minute calls where a single-request gold_full
transcript is not trustworthy because the audio token budget leaves too little
completion room. It replaces gold_full artifacts with a chunked reference
transcript, recomputes candidate-vs-reference metrics for selected scenarios,
and rebuilds the root-level scoreboards/review bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from vllm_plugin.experiments.run_chunking_variant_experiment import (
        _build_chunk_starts,
        _compare_to_gold,
        _compute_seam_metrics,
    )
    from vllm_plugin.experiments.run_finalist_validation import (
        _build_root_review_bundle,
        _build_variant_score_rows,
        _scenario_name,
        _write_csv,
        _write_md,
    )
except ModuleNotFoundError:
    from run_chunking_variant_experiment import (
        _build_chunk_starts,
        _compare_to_gold,
        _compute_seam_metrics,
    )
    from run_finalist_validation import (
        _build_root_review_bundle,
        _build_variant_score_rows,
        _scenario_name,
        _write_csv,
        _write_md,
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    out: list[dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("Start")
        end = seg.get("End")
        content = seg.get("Content")
        if start is None or end is None or content is None:
            continue
        out.append(
            {
                "Start": float(start),
                "End": float(end),
                "Speaker": seg.get("Speaker"),
                "Content": str(content),
            }
        )
    return out


def _copy_reference_as_gold(
    *,
    merged_variant_dir: Path,
    reference_artifact_dir: Path,
    file_id: str,
    reference_scenario: str,
) -> None:
    file_slug = _slugify(file_id)
    src_dir = reference_artifact_dir / "transcripts" / file_slug
    dst_dir = merged_variant_dir / "transcripts" / file_slug
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_json = src_dir / f"{_slugify(reference_scenario)}.json"
    src_txt = src_dir / f"{_slugify(reference_scenario)}.txt"
    if not src_json.exists() or not src_txt.exists():
        raise FileNotFoundError(f"Missing reference transcript for {file_id}: {src_json}")

    payload = json.loads(src_json.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    payload["metadata"] = {
        **metadata,
        "reference_source": str(reference_artifact_dir),
        "reference_scenario": reference_scenario,
        "refreshed_at": _utc_now_iso(),
    }
    (dst_dir / "gold_full.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(src_txt, dst_dir / "gold_full.txt")


def _refresh_variant_rows(
    *,
    variant_dir: Path,
    file_ids: set[str],
    scenario: str,
    chunk_minutes: int,
    overlap_seconds: float,
    seam_window_seconds: float,
    quality_threshold: float,
    reference_artifact_dir: Path,
    reference_scenario: str,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(variant_dir / "runs.jsonl")
    updated: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        file_id = str(row.get("file_id"))
        row_scenario = str(row.get("scenario"))
        if file_id in file_ids and row_scenario == "gold_full":
            row["gold_source"] = "reference_chunked"
            row["generated_at"] = _utc_now_iso()
        if file_id in file_ids and row_scenario == scenario and bool(row.get("success")):
            file_slug = _slugify(file_id)
            gold_json = variant_dir / "transcripts" / file_slug / "gold_full.json"
            cand_json = variant_dir / "transcripts" / file_slug / f"{_slugify(scenario)}.json"
            if gold_json.exists() and cand_json.exists():
                gold_segments = _load_segments(gold_json)
                cand_segments = _load_segments(cand_json)
                compare = _compare_to_gold(gold_segments, cand_segments)
                duration = float(row.get("duration_sec") or 0.0)
                chunk_seconds = float(chunk_minutes * 60)
                seam_boundaries = _build_chunk_starts(duration, chunk_seconds, overlap_seconds)[1:]
                seam_metrics = _compute_seam_metrics(
                    gold_segments=gold_segments,
                    cand_segments=cand_segments,
                    seam_boundaries=seam_boundaries,
                    seam_window_sec=seam_window_seconds,
                )
                row["word_drift"] = compare.get("word_drift")
                row["char_drift"] = compare.get("char_drift")
                row["segment_count_delta"] = compare.get("segment_count_delta")
                row["boundary_mae_sec"] = compare.get("boundary_mae_sec")
                row["seam_word_drift"] = seam_metrics.get("seam_word_drift")
                row["seam_boundary_mae_sec"] = seam_metrics.get("seam_boundary_mae_sec")
                row["quality_pass"] = (
                    compare.get("word_drift") is not None
                    and float(compare["word_drift"]) <= quality_threshold
                )
                row["generated_at"] = _utc_now_iso()
        updated.append(row)

    fieldnames = sorted({key for row in updated for key in row.keys()})
    summary_path = variant_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated:
            writer.writerow(row)
    _write_jsonl(variant_dir / "runs.jsonl", updated)
    return updated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh long-tail artifacts with chunked reference gold.")
    parser.add_argument("--run-root", required=True, help="Existing finalist long-tail run root.")
    parser.add_argument(
        "--reference-artifact-dir",
        required=True,
        help="Artifact directory containing reference chunk transcripts.",
    )
    parser.add_argument(
        "--reference-scenario",
        default="chunk_10m_overlap_context",
        help="Scenario name inside the reference artifact to use as gold.",
    )
    parser.add_argument(
        "--file-ids",
        nargs="+",
        required=True,
        help="File IDs whose gold_full transcripts should be replaced.",
    )
    parser.add_argument(
        "--scenario",
        default="chunk_30m_overlap_context",
        help="Candidate scenario in the finalist run to recompute metrics for.",
    )
    parser.add_argument("--chunk-minutes", type=int, default=30)
    parser.add_argument("--overlap-seconds", type=float, default=30.0)
    parser.add_argument("--seam-window-seconds", type=float, default=60.0)
    parser.add_argument("--quality-threshold", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    reference_artifact_dir = Path(args.reference_artifact_dir).resolve()
    file_ids = {str(value) for value in args.file_ids}
    scenario = str(args.scenario)

    merged_root = run_root / "merged"
    if not merged_root.exists():
        raise FileNotFoundError(f"Merged artifact root not found: {merged_root}")

    variant_runs: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    variants = sorted(path.name for path in merged_root.iterdir() if path.is_dir())
    for variant in variants:
        variant_dir = merged_root / variant
        for file_id in file_ids:
            _copy_reference_as_gold(
                merged_variant_dir=variant_dir,
                reference_artifact_dir=reference_artifact_dir,
                file_id=file_id,
                reference_scenario=args.reference_scenario,
            )
        rows = _refresh_variant_rows(
            variant_dir=variant_dir,
            file_ids=file_ids,
            scenario=scenario,
            chunk_minutes=args.chunk_minutes,
            overlap_seconds=args.overlap_seconds,
            seam_window_seconds=args.seam_window_seconds,
            quality_threshold=args.quality_threshold,
            reference_artifact_dir=reference_artifact_dir,
            reference_scenario=args.reference_scenario,
        )
        variant_runs[variant] = (variant_dir, rows)

    scoreboard_rows = _build_variant_score_rows(variant_runs, scenarios=[scenario])
    scoreboard_csv = run_root / "variant_scoreboard.csv"
    scoreboard_md = run_root / "variant_scoreboard.md"
    _write_csv(scoreboard_csv, scoreboard_rows)
    _write_md(scoreboard_md, "Finalist Long-Call Metric Scoreboard", scoreboard_rows)
    _build_root_review_bundle(run_root, variant_runs, scenarios=[scenario])

    refresh_report = {
        "generated_at": _utc_now_iso(),
        "run_root": str(run_root),
        "reference_artifact_dir": str(reference_artifact_dir),
        "reference_scenario": args.reference_scenario,
        "file_ids": sorted(file_ids),
        "scenario": scenario,
        "paths": {
            "scoreboard_csv": str(scoreboard_csv),
            "scoreboard_md": str(scoreboard_md),
            "expert_review_template_json": str(run_root / "expert_review" / "review_template.json"),
        },
    }
    (run_root / "reference_gold_refresh.json").write_text(
        json.dumps(refresh_report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print("[INFO] Reference-gold refresh complete.")
    print(f"[INFO] Run root: {run_root}")
    print(f"[INFO] Reference artifact: {reference_artifact_dir}")


if __name__ == "__main__":
    main()
