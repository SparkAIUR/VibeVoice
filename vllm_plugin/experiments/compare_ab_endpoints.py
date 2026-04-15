#!/usr/bin/env python3
"""
Compare Spark-vs-Production experiment artifacts using production gold transcripts.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from vllm_plugin.experiments.expert_review import make_review_item, write_review_bundle
except ModuleNotFoundError:
    from expert_review import make_review_item, write_review_bundle


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _join_segment_content(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(seg.get("Content", "")).strip() for seg in segments if seg.get("Content"))


def _compare_to_gold(
    gold_segments: list[dict[str, Any]],
    cand_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_text = _join_segment_content(gold_segments)
    cand_text = _join_segment_content(cand_segments)

    char_ratio = difflib.SequenceMatcher(
        a=_normalize_text(gold_text), b=_normalize_text(cand_text)
    ).ratio()
    gold_words = _normalize_text(gold_text).split()
    cand_words = _normalize_text(cand_text).split()
    word_ratio = difflib.SequenceMatcher(a=gold_words, b=cand_words).ratio()

    pair_count = min(len(gold_segments), len(cand_segments))
    if pair_count > 0:
        start_errors = [
            abs(float(cand_segments[i]["Start"]) - float(gold_segments[i]["Start"]))
            for i in range(pair_count)
        ]
        end_errors = [
            abs(float(cand_segments[i]["End"]) - float(gold_segments[i]["End"]))
            for i in range(pair_count)
        ]
        boundary_mae = statistics.mean(start_errors + end_errors)
    else:
        boundary_mae = None

    return {
        "char_drift": round(1.0 - char_ratio, 6),
        "word_drift": round(1.0 - word_ratio, 6),
        "segment_count_delta": len(cand_segments) - len(gold_segments),
        "boundary_mae_sec": None if boundary_mae is None else round(boundary_mae, 6),
        "gold_char_len": len(gold_text),
        "cand_char_len": len(cand_text),
    }


def _extract_diff_snippets(gold_text: str, cand_text: str, max_snippets: int = 3) -> list[str]:
    gold_words = _normalize_text(gold_text).split()
    cand_words = _normalize_text(cand_text).split()
    matcher = difflib.SequenceMatcher(a=gold_words, b=cand_words)
    snippets: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        gold_excerpt = " ".join(gold_words[max(0, i1 - 8) : min(len(gold_words), i2 + 8)])
        cand_excerpt = " ".join(cand_words[max(0, j1 - 8) : min(len(cand_words), j2 + 8)])
        snippets.append(
            f"- Change type: `{tag}`\n"
            f"  - Gold excerpt: `{gold_excerpt[:300]}`\n"
            f"  - Candidate excerpt: `{cand_excerpt[:300]}`"
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def _read_runs(runs_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_path.exists():
        return rows
    with runs_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_transcript_segments(artifact_dir: Path, file_id: str, scenario: str) -> list[dict[str, Any]]:
    transcript_path = (
        artifact_dir / "transcripts" / _slugify(file_id) / f"{_slugify(scenario)}.json"
    )
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    return payload.get("segments", [])


def _collect_prod_gold_map(prod_dir: Path, prod_runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    gold_map: dict[str, list[dict[str, Any]]] = {}
    for row in prod_runs:
        if row.get("scenario") != "gold_full":
            continue
        if not row.get("success"):
            continue
        file_id = str(row["file_id"])
        gold_map[file_id] = _load_transcript_segments(prod_dir, file_id, "gold_full")
    return gold_map


def _is_throughput_row(row: dict[str, Any]) -> bool:
    scenario = str(row.get("scenario", ""))
    return scenario.startswith("throughput_workers_")


def _collect_quality_rows(
    endpoint: str,
    artifact_dir: Path,
    runs: list[dict[str, Any]],
    prod_gold: dict[str, list[dict[str, Any]]],
    quality_threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in runs:
        scenario = str(row.get("scenario", ""))
        file_id = str(row.get("file_id", ""))
        if scenario == "gold_full":
            continue
        if _is_throughput_row(row):
            continue
        if file_id == "__aggregate__":
            continue

        gold_segments = prod_gold.get(file_id)
        success = bool(row.get("success"))
        candidate_segments: list[dict[str, Any]] = []
        error = row.get("error")
        compare = {
            "char_drift": None,
            "word_drift": None,
            "segment_count_delta": None,
            "boundary_mae_sec": None,
        }
        if gold_segments is None:
            success = False
            error = "Missing production gold transcript for file."
        elif success:
            try:
                candidate_segments = _load_transcript_segments(artifact_dir, file_id, scenario)
                compare = _compare_to_gold(gold_segments, candidate_segments)
            except Exception as exc:  # noqa: BLE001
                success = False
                error = f"Failed to read candidate transcript: {exc}"

        out.append(
            {
                "endpoint": endpoint,
                "artifact_dir": str(artifact_dir),
                "file_id": file_id,
                "file_path": row.get("file_path"),
                "scenario": scenario,
                "success": success,
                "error": error,
                "duration_sec": row.get("duration_sec"),
                "wall_time_sec": row.get("wall_time_sec"),
                "rtf": row.get("rtf"),
                "chunk_count": row.get("chunk_count"),
                "chunk_failures": row.get("chunk_failures"),
                "retry_count": row.get("retry_count"),
                "chunk_latency_p95_sec": row.get("chunk_latency_p95_sec"),
                "word_drift": compare.get("word_drift"),
                "char_drift": compare.get("char_drift"),
                "segment_count_delta": compare.get("segment_count_delta"),
                "boundary_mae_sec": compare.get("boundary_mae_sec"),
                "quality_pass": (
                    compare.get("word_drift") is not None
                    and float(compare["word_drift"]) <= quality_threshold
                ),
                "generated_at": _utc_now_iso(),
            }
        )
    return out


def _collect_throughput_rows(
    endpoint: str,
    artifact_dir: Path,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in runs:
        if not _is_throughput_row(row):
            continue
        out.append(
            {
                "endpoint": endpoint,
                "artifact_dir": str(artifact_dir),
                "scenario": row.get("scenario"),
                "worker_count": row.get("worker_count"),
                "success": row.get("success"),
                "duration_sec": row.get("duration_sec"),
                "wall_time_sec": row.get("wall_time_sec"),
                "rtf": row.get("rtf"),
                "chunk_count": row.get("chunk_count"),
                "chunk_failures": row.get("chunk_failures"),
                "retry_count": row.get("retry_count"),
                "throughput_files_per_hour": row.get("throughput_files_per_hour"),
                "throughput_audio_hours_per_hour": row.get("throughput_audio_hours_per_hour"),
                "generated_at": _utc_now_iso(),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize_quality_md(
    path: Path,
    quality_rows: list[dict[str, Any]],
    threshold: float,
) -> str:
    lines = [
        "# A/B Quality Summary",
        "",
        f"- Quality threshold: `{threshold:.4f}`",
        "",
        "| Endpoint | Scenario | Runs | Success | Pass | Mean Word Drift | Mean RTF |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in quality_rows:
        grouped.setdefault((str(row["endpoint"]), str(row["scenario"])), []).append(row)

    for endpoint, scenario in sorted(grouped):
        rows = grouped[(endpoint, scenario)]
        ok_rows = [r for r in rows if r.get("success")]
        pass_rows = [r for r in ok_rows if r.get("quality_pass")]
        word_vals = [float(r["word_drift"]) for r in ok_rows if r.get("word_drift") is not None]
        rtf_vals = [float(r["rtf"]) for r in ok_rows if r.get("rtf") is not None]
        lines.append(
            "| {endpoint} | {scenario} | {runs} | {ok} | {passed} | {word} | {rtf} |".format(
                endpoint=endpoint,
                scenario=scenario,
                runs=len(rows),
                ok=len(ok_rows),
                passed=len(pass_rows),
                word=f"{statistics.mean(word_vals):.4f}" if word_vals else "NA",
                rtf=f"{statistics.mean(rtf_vals):.4f}" if rtf_vals else "NA",
            )
        )

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _summarize_throughput_md(path: Path, throughput_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# A/B Throughput Summary",
        "",
        "| Endpoint | Scenario | Workers | Success | Audio Hours/Hour | Files/Hour | RTF |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        throughput_rows,
        key=lambda r: (str(r.get("endpoint")), int(r.get("worker_count") or 0)),
    ):
        lines.append(
            "| {endpoint} | {scenario} | {workers} | {ok} | {ahph} | {fph} | {rtf} |".format(
                endpoint=row.get("endpoint"),
                scenario=row.get("scenario"),
                workers=row.get("worker_count"),
                ok=row.get("success"),
                ahph=(
                    f"{float(row['throughput_audio_hours_per_hour']):.4f}"
                    if row.get("throughput_audio_hours_per_hour") is not None
                    else "NA"
                ),
                fph=(
                    f"{float(row['throughput_files_per_hour']):.2f}"
                    if row.get("throughput_files_per_hour") is not None
                    else "NA"
                ),
                rtf=f"{float(row['rtf']):.4f}" if row.get("rtf") is not None else "NA",
            )
        )
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _build_manual_review(
    path: Path,
    quality_rows: list[dict[str, Any]],
    prod_dir: Path,
    spark_dir: Path,
    threshold: float,
) -> None:
    lines: list[str] = ["# A/B Manual Diff Review", ""]
    flagged = [
        row
        for row in quality_rows
        if row.get("success")
        and row.get("word_drift") is not None
        and (float(row["word_drift"]) > threshold or float(row["word_drift"]) > 0.01)
    ]
    if not flagged:
        lines.append("No scenarios exceeded review thresholds.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    endpoint_to_dir = {"production": prod_dir, "spark": spark_dir}
    for row in flagged:
        endpoint = str(row["endpoint"])
        file_id = str(row["file_id"])
        scenario = str(row["scenario"])
        endpoint_dir = endpoint_to_dir.get(endpoint)
        if endpoint_dir is None:
            continue
        gold_segments = _load_transcript_segments(prod_dir, file_id, "gold_full")
        cand_segments = _load_transcript_segments(endpoint_dir, file_id, scenario)
        snippets = _extract_diff_snippets(
            _join_segment_content(gold_segments), _join_segment_content(cand_segments)
        )
        lines.append(f"## {endpoint} :: {file_id} :: {scenario}")
        lines.append(f"- Word drift: {float(row['word_drift']):.4f}")
        lines.append(f"- Char drift: {float(row['char_drift']):.4f}")
        if snippets:
            lines.extend(snippets)
        else:
            lines.append("- No diff snippets could be extracted.")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_expert_review_bundle(
    output_dir: Path,
    quality_rows: list[dict[str, Any]],
    prod_dir: Path,
    spark_dir: Path,
) -> None:
    items: list[dict[str, Any]] = []
    endpoint_to_dir = {"production": prod_dir, "spark": spark_dir}
    for row in quality_rows:
        if not row.get("success"):
            continue
        endpoint = str(row.get("endpoint"))
        file_id = str(row.get("file_id"))
        scenario = str(row.get("scenario"))
        endpoint_dir = endpoint_to_dir.get(endpoint)
        if endpoint_dir is None:
            continue
        gold_txt_path = prod_dir / "transcripts" / _slugify(file_id) / "gold_full.txt"
        cand_txt_path = endpoint_dir / "transcripts" / _slugify(file_id) / f"{_slugify(scenario)}.txt"
        gold_segments = _load_transcript_segments(prod_dir, file_id, "gold_full")
        cand_segments = _load_transcript_segments(endpoint_dir, file_id, scenario)
        diff_snippets = _extract_diff_snippets(
            _join_segment_content(gold_segments),
            _join_segment_content(cand_segments),
        )
        items.append(
            make_review_item(
                comparison={
                    "endpoint": endpoint,
                    "file_id": file_id,
                    "scenario": scenario,
                },
                gold_txt_path=gold_txt_path,
                candidate_txt_path=cand_txt_path,
                metrics={
                    key: row.get(key)
                    for key in (
                        "word_drift",
                        "char_drift",
                        "boundary_mae_sec",
                        "rtf",
                        "wall_time_sec",
                    )
                },
                diff_snippets=diff_snippets,
            )
        )

    write_review_bundle(
        output_dir=output_dir,
        title="A/B Endpoint Expert Review Template",
        items=items,
        group_by_fields=("endpoint", "scenario"),
        source_kind="ab_compare",
        metadata={
            "production_artifact_dir": str(prod_dir),
            "spark_artifact_dir": str(spark_dir),
            "review_scope": "All successful quality rows compared against production gold.",
        },
    )


def _extract_base_scenario(throughput_scenario_name: str) -> str | None:
    match = re.match(r"^throughput_workers_\d+_(.+)$", throughput_scenario_name)
    return match.group(1) if match else None


def _recommend_prod_profile(
    quality_rows: list[dict[str, Any]],
    throughput_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prod_quality = [
        row
        for row in quality_rows
        if row.get("endpoint") == "production"
        and row.get("success")
        and row.get("word_drift") is not None
    ]
    if not prod_quality:
        return {
            "recommended_scenario": None,
            "recommended_worker_count": None,
            "reason": "No successful production quality rows found.",
        }

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in prod_quality:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    avg_quality = {
        scenario: statistics.mean(float(r["word_drift"]) for r in rows)
        for scenario, rows in by_scenario.items()
    }
    best_drift = min(avg_quality.values())
    allowed = {s for s, v in avg_quality.items() if v <= best_drift + 0.03}

    prod_tp = [r for r in throughput_rows if r.get("endpoint") == "production" and r.get("success")]
    scored_tp: list[dict[str, Any]] = []
    for row in prod_tp:
        scenario_name = str(row.get("scenario"))
        base_scenario = _extract_base_scenario(scenario_name)
        if base_scenario not in allowed:
            continue
        score = (
            float(row["throughput_audio_hours_per_hour"])
            if row.get("throughput_audio_hours_per_hour") is not None
            else 0.0
        )
        scored_tp.append(
            {
                "base_scenario": base_scenario,
                "scenario": scenario_name,
                "worker_count": row.get("worker_count"),
                "throughput_audio_hours_per_hour": row.get("throughput_audio_hours_per_hour"),
                "throughput_files_per_hour": row.get("throughput_files_per_hour"),
                "score": score,
            }
        )

    if scored_tp:
        best_tp = max(scored_tp, key=lambda r: (float(r["score"]), int(r.get("worker_count") or 0)))
        return {
            "recommended_scenario": best_tp["base_scenario"],
            "recommended_worker_count": best_tp["worker_count"],
            "recommended_throughput_audio_hours_per_hour": best_tp["throughput_audio_hours_per_hour"],
            "reason": "Highest throughput among scenarios within +0.03 word drift of best quality.",
        }

    best_quality_scenario = min(avg_quality.items(), key=lambda kv: kv[1])[0]
    return {
        "recommended_scenario": best_quality_scenario,
        "recommended_worker_count": None,
        "recommended_throughput_audio_hours_per_hour": None,
        "reason": "Throughput rows unavailable; selected best-quality production scenario.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Spark vs Production experiment artifacts using production gold baseline."
    )
    parser.add_argument("--prod-dir", required=True, help="Production artifact directory.")
    parser.add_argument("--spark-dir", required=True, help="Spark artifact directory.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/chunking_experiment/ab_compare",
        help="Output root directory for A/B summaries.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.02,
        help="Word drift threshold for quality pass/fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prod_dir = Path(args.prod_dir).resolve()
    spark_dir = Path(args.spark_dir).resolve()

    if not prod_dir.exists():
        raise FileNotFoundError(f"Production artifact directory not found: {prod_dir}")
    if not spark_dir.exists():
        raise FileNotFoundError(f"Spark artifact directory not found: {spark_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    prod_runs = _read_runs(prod_dir / "runs.jsonl")
    spark_runs = _read_runs(spark_dir / "runs.jsonl")
    prod_gold = _collect_prod_gold_map(prod_dir, prod_runs)

    quality_rows = []
    quality_rows.extend(
        _collect_quality_rows(
            endpoint="production",
            artifact_dir=prod_dir,
            runs=prod_runs,
            prod_gold=prod_gold,
            quality_threshold=args.quality_threshold,
        )
    )
    quality_rows.extend(
        _collect_quality_rows(
            endpoint="spark",
            artifact_dir=spark_dir,
            runs=spark_runs,
            prod_gold=prod_gold,
            quality_threshold=args.quality_threshold,
        )
    )

    throughput_rows = []
    throughput_rows.extend(_collect_throughput_rows("production", prod_dir, prod_runs))
    throughput_rows.extend(_collect_throughput_rows("spark", spark_dir, spark_runs))

    quality_csv = output_dir / "ab_quality_summary.csv"
    quality_md = output_dir / "ab_quality_summary.md"
    review_md = output_dir / "ab_manual_review.md"
    throughput_csv = output_dir / "ab_throughput_summary.csv"
    throughput_md = output_dir / "ab_throughput_summary.md"
    recommendation_json = output_dir / "ab_recommendation.json"

    _write_csv(quality_csv, quality_rows)
    summary_text = _summarize_quality_md(quality_md, quality_rows, args.quality_threshold)
    _build_manual_review(review_md, quality_rows, prod_dir, spark_dir, args.quality_threshold)
    _build_expert_review_bundle(output_dir, quality_rows, prod_dir, spark_dir)
    if throughput_rows:
        _write_csv(throughput_csv, throughput_rows)
        _summarize_throughput_md(throughput_md, throughput_rows)

    recommendation = _recommend_prod_profile(quality_rows, throughput_rows)
    recommendation_json.write_text(json.dumps(recommendation, ensure_ascii=True, indent=2), encoding="utf-8")

    print("[INFO] A/B comparison complete.")
    print(f"[INFO] Production dir: {prod_dir}")
    print(f"[INFO] Spark dir:      {spark_dir}")
    print(f"[INFO] Output dir:     {output_dir}")
    print("[INFO] Recommendation:")
    print(json.dumps(recommendation, ensure_ascii=True, indent=2))
    print("[INFO] Quality summary:")
    print(summary_text)


if __name__ == "__main__":
    main()
