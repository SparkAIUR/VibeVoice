#!/usr/bin/env python3
"""
Run finalist-only long-call validation for selected chunking variants.

This script:
- runs baseline first to generate/reuse gold transcripts
- runs the remaining finalist variants against the same manifest
- writes a cross-variant metric scoreboard
- emits a root-level expert review bundle for subjective reranking
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from vllm_plugin.experiments.expert_review import make_review_item, write_review_bundle
except ModuleNotFoundError:
    from expert_review import make_review_item, write_review_bundle


DEFAULT_VARIANTS = [
    "baseline_overlap_context",
    "idea3_shifted_consensus",
    "idea5_dynamic_lexicon",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


def _scenario_name(chunk_minutes: int) -> str:
    return f"chunk_{chunk_minutes}m_overlap_context"


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_summary_rows(summary_csv_path: Path) -> list[dict[str, Any]]:
    if not summary_csv_path.exists():
        return []
    with summary_csv_path.open("r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _run_command_capture(cmd: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = ""
    with log_path.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to capture subprocess output.")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()
            if "Artifacts directory:" in line:
                artifact_dir = line.split("Artifacts directory:", 1)[1].strip()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
    if not artifact_dir:
        raise RuntimeError(f"Could not parse artifact directory from log: {log_path}")
    return artifact_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            "| Variant | Scenario | Files | Mean word drift | Mean seam drift | Mean char drift | Mean RTF | Mean wall time |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {variant} | {scenario} | {count} | {word} | {seam} | {char} | {rtf} | {wall} |".format(
                variant=row["variant"],
                scenario=row["scenario"],
                count=row["quality_runs"],
                word=f"{float(row['mean_word_drift']):.4f}" if row.get("mean_word_drift") is not None else "NA",
                seam=f"{float(row['mean_seam_word_drift']):.4f}" if row.get("mean_seam_word_drift") is not None else "NA",
                char=f"{float(row['mean_char_drift']):.4f}" if row.get("mean_char_drift") is not None else "NA",
                rtf=f"{float(row['mean_rtf']):.4f}" if row.get("mean_rtf") is not None else "NA",
                wall=f"{float(row['mean_wall_time_sec']):.2f}" if row.get("mean_wall_time_sec") is not None else "NA",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


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
            f"change={tag}; gold=`{gold_excerpt[:220]}`; candidate=`{cand_excerpt[:220]}`"
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def _build_variant_score_rows(
    variant_runs: dict[str, tuple[Path, list[dict[str, Any]]]],
    scenarios: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant, (_, rows) in variant_runs.items():
        for scenario in scenarios:
            target = [
                row
                for row in rows
                if str(row.get("scenario")) == scenario
                and str(row.get("file_id")) != "__aggregate__"
                and row.get("success")
                and _safe_float(row.get("word_drift")) is not None
            ]
            if not target:
                continue
            out.append(
                {
                    "variant": variant,
                    "scenario": scenario,
                    "quality_runs": len(target),
                    "mean_word_drift": round(statistics.mean(float(r["word_drift"]) for r in target), 6),
                    "mean_seam_word_drift": round(
                        statistics.mean(
                            float(r["seam_word_drift"])
                            for r in target
                            if r.get("seam_word_drift") not in (None, "")
                        ),
                        6,
                    )
                    if any(r.get("seam_word_drift") not in (None, "") for r in target)
                    else None,
                    "mean_char_drift": round(
                        statistics.mean(float(r["char_drift"]) for r in target if r.get("char_drift") not in (None, "")),
                        6,
                    ),
                    "mean_rtf": round(
                        statistics.mean(float(r["rtf"]) for r in target if r.get("rtf") not in (None, "")),
                        6,
                    ),
                    "mean_wall_time_sec": round(
                        statistics.mean(
                            float(r["wall_time_sec"]) for r in target if r.get("wall_time_sec") not in (None, "")
                        ),
                        6,
                    ),
                }
            )
    out.sort(
        key=lambda row: (
            row["scenario"],
            float(row.get("mean_word_drift") or 999.0),
            float(row.get("mean_rtf") or 999.0),
        )
    )
    return out


def _build_root_review_bundle(
    root: Path,
    variant_runs: dict[str, tuple[Path, list[dict[str, Any]]]],
    scenarios: list[str],
) -> None:
    items: list[dict[str, Any]] = []
    for variant, (artifact_dir, rows) in variant_runs.items():
        for row in rows:
            scenario = str(row.get("scenario"))
            if scenario not in scenarios:
                continue
            if str(row.get("file_id")) == "__aggregate__":
                continue
            if not row.get("success"):
                continue
            file_id = str(row["file_id"])
            file_slug = _slugify(file_id)
            gold_txt_path = artifact_dir / "transcripts" / file_slug / "gold_full.txt"
            cand_txt_path = artifact_dir / "transcripts" / file_slug / f"{_slugify(scenario)}.txt"
            if not gold_txt_path.exists() or not cand_txt_path.exists():
                continue
            gold_text = gold_txt_path.read_text(encoding="utf-8")
            cand_text = cand_txt_path.read_text(encoding="utf-8")
            items.append(
                make_review_item(
                    comparison={
                        "variant": variant,
                        "file_id": file_id,
                        "scenario": scenario,
                    },
                    gold_txt_path=gold_txt_path,
                    candidate_txt_path=cand_txt_path,
                    metrics={
                        key: row.get(key)
                        for key in (
                            "word_drift",
                            "seam_word_drift",
                            "char_drift",
                            "boundary_mae_sec",
                            "chunk_latency_p95_sec",
                            "wall_time_sec",
                            "rtf",
                        )
                    },
                    diff_snippets=_extract_diff_snippets(gold_text, cand_text),
                )
            )
    write_review_bundle(
        output_dir=root,
        title="Finalist Long-Call Validation Expert Review Template",
        items=items,
        group_by_fields=("variant", "scenario"),
        source_kind="finalist_validation",
        metadata={
            "review_scope": "All successful finalist transcripts for the selected long-tail manifest.",
        },
    )


def _run_variant(
    root: Path,
    args: argparse.Namespace,
    variant: str,
    *,
    gold_artifact_dir: Path | None,
    skip_gold: bool,
) -> Path:
    run_output_dir = root / "runs" / variant
    log_path = root / "logs" / f"{variant}.log"
    cmd = [
        "python3",
        "-u",
        "vllm_plugin/experiments/run_chunking_variant_experiment.py",
        "--manifest",
        str(Path(args.manifest).resolve()),
        "--output-dir",
        str(run_output_dir),
        "--api-url",
        args.api_url,
        "--model",
        args.model,
        "--variant",
        variant,
        "--chunk-minutes",
        *[str(value) for value in args.chunk_minutes],
        "--scenario-modes",
        "overlap_context",
        "--overlap-seconds",
        str(args.overlap_seconds),
        "--context-tail-chars",
        str(args.context_tail_chars),
        "--quality-threshold",
        str(args.quality_threshold),
        "--max-tokens",
        str(args.max_tokens),
        "--max-tokens-gold",
        str(args.max_tokens_gold),
        "--max-tokens-chunk",
        str(args.max_tokens_chunk),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-retries",
        str(args.max_retries),
    ]
    if args.run_throughput:
        cmd.extend(
            [
                "--run-throughput",
                "--throughput-scenario",
                _scenario_name(args.chunk_minutes[0]),
                "--throughput-workers",
                *[str(value) for value in args.throughput_workers],
            ]
        )
    if gold_artifact_dir:
        cmd.extend(["--gold-artifact-dir", str(gold_artifact_dir)])
    if skip_gold:
        cmd.append("--skip-gold")
    artifact_dir = _run_command_capture(cmd, log_path)
    return Path(artifact_dir).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finalist-only long-call validation.")
    parser.add_argument(
        "--manifest",
        default="vllm_plugin/experiments/manifests/endpoint_02172026_long_tail_6.json",
        help="Manifest path for long-call validation set.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/chunking_experiment/endpoint_finalists_long_tail",
        help="Root output directory.",
    )
    parser.add_argument("--api-url", required=True, help="vLLM OpenAI-compatible endpoint URL.")
    parser.add_argument("--model", default="vibevoice", help="Served model name.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="Finalist variants to run. Baseline must be included.",
    )
    parser.add_argument(
        "--chunk-minutes",
        nargs="+",
        type=int,
        default=[30],
        help="Chunk sizes to evaluate. Default keeps the long-call run focused on 30m overlap-context.",
    )
    parser.add_argument("--overlap-seconds", type=float, default=30.0)
    parser.add_argument("--context-tail-chars", type=int, default=800)
    parser.add_argument("--quality-threshold", type=float, default=0.02)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens-gold", type=int, default=32768)
    parser.add_argument("--max-tokens-chunk", type=int, default=24576)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--run-throughput", action="store_true", help="Run throughput matrix for each finalist.")
    parser.add_argument("--throughput-workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(
        "--baseline-gold-artifact-dir",
        default=None,
        help="Optional artifact directory to reuse baseline gold_full transcripts.",
    )
    parser.add_argument(
        "--baseline-skip-gold",
        action="store_true",
        help="When set, baseline run skips live gold generation and requires cached baseline gold.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variants = [str(value) for value in args.variants]
    if "baseline_overlap_context" not in variants:
        raise ValueError("Variants must include baseline_overlap_context.")
    if variants[0] != "baseline_overlap_context":
        variants = ["baseline_overlap_context"] + [v for v in variants if v != "baseline_overlap_context"]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_dir).resolve() / timestamp
    root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    (root / "manifest_used.json").write_text(
        json.dumps(
            {
                "generated_at": _utc_now_iso(),
                "manifest_source": str(manifest_path),
                "variants": variants,
                "chunk_minutes": args.chunk_minutes,
                "api_url": args.api_url,
                "files": manifest_payload.get("files", manifest_payload),
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    variant_runs: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    baseline_artifact: Path | None = None
    baseline_gold_artifact = (
        Path(args.baseline_gold_artifact_dir).resolve()
        if args.baseline_gold_artifact_dir
        else None
    )

    for index, variant in enumerate(variants):
        print(f"[INFO] Running finalist variant: {variant}")
        use_gold_cache = baseline_artifact if baseline_artifact else baseline_gold_artifact
        artifact_dir = _run_variant(
            root,
            args,
            variant,
            gold_artifact_dir=use_gold_cache,
            skip_gold=(index > 0) or bool(index == 0 and args.baseline_skip_gold),
        )
        rows = _read_summary_rows(artifact_dir / "summary.csv")
        variant_runs[variant] = (artifact_dir, rows)
        if variant == "baseline_overlap_context":
            baseline_artifact = artifact_dir

    scenarios = [_scenario_name(value) for value in args.chunk_minutes]
    scoreboard_rows = _build_variant_score_rows(variant_runs, scenarios=scenarios)
    scoreboard_csv = root / "variant_scoreboard.csv"
    scoreboard_md = root / "variant_scoreboard.md"
    _write_csv(scoreboard_csv, scoreboard_rows)
    _write_md(scoreboard_md, "Finalist Long-Call Metric Scoreboard", scoreboard_rows)
    _build_root_review_bundle(root, variant_runs, scenarios=scenarios)

    recommendation = {
        "generated_at": _utc_now_iso(),
        "api_url": args.api_url,
        "variants": variants,
        "scenarios": scenarios,
        "paths": {
            "scoreboard_csv": str(scoreboard_csv),
            "scoreboard_md": str(scoreboard_md),
            "expert_review_template_json": str(root / "expert_review" / "review_template.json"),
        },
    }
    recommendation_path = root / "run_manifest.json"
    recommendation_path.write_text(json.dumps(recommendation, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print("[INFO] Finalist validation complete.")
    print(f"[INFO] Root output: {root}")
    print(f"[INFO] Scoreboard: {scoreboard_csv}")


if __name__ == "__main__":
    main()
