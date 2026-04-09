#!/usr/bin/env python3
"""
Orchestrate two-phase variant experiments for long-audio ASR chunking.

Phase 1:
- Baseline + five single ideas on stratified pilot set
- Greedy combinations from the best single idea

Phase 2:
- Baseline + Phase-1 finalists on full set

Outputs:
- variant_scoreboard.csv/.md
- combo_scoreboard.csv/.md
- final_recommendation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANT_ALIASES: dict[str, set[str]] = {
    "baseline_overlap_context": set(),
    "idea1_sentence_seam": {"sentence_seam"},
    "idea2_seam_micro_redo": {"seam_micro_redo"},
    "idea3_shifted_consensus": {"shifted_consensus"},
    "idea4_silence_aligned_boundaries": {"silence_aligned_boundaries"},
    "idea5_dynamic_lexicon": {"dynamic_lexicon"},
}

IDEA_ORDER = [
    "idea1_sentence_seam",
    "idea2_seam_micro_redo",
    "idea3_shifted_consensus",
    "idea4_silence_aligned_boundaries",
    "idea5_dynamic_lexicon",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_grid(max_workers: int) -> list[int]:
    base = [1, 2, 4, 8, 12, 16]
    return [w for w in base if w <= max_workers]


def _scenario_name(chunk_minutes: int) -> str:
    return f"chunk_{chunk_minutes}m_overlap_context"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
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


def _run_variant_experiment(
    phase_root: Path,
    label: str,
    variant_spec: str,
    manifest_path: Path,
    args: argparse.Namespace,
    throughput_workers: list[int],
    scenario_name: str,
    gold_artifact_dir: Path | None,
    skip_gold: bool,
) -> Path:
    run_output_dir = phase_root / "runs" / label
    log_path = phase_root / "logs" / f"{label}.log"
    cmd = [
        "python3",
        "-u",
        "vllm_plugin/experiments/run_chunking_variant_experiment.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(run_output_dir),
        "--api-url",
        args.api_url,
        "--model",
        args.model,
        "--variant",
        variant_spec,
        "--chunk-minutes",
        str(args.chunk_minutes),
        "--scenario-modes",
        "overlap_context",
        "--overlap-seconds",
        str(args.overlap_seconds),
        "--context-tail-chars",
        str(args.context_tail_chars),
        "--seam-window-seconds",
        str(args.seam_window_seconds),
        "--micro-redo-window-seconds",
        str(args.micro_redo_window_seconds),
        "--shifted-offset-ratio",
        str(args.shifted_offset_ratio),
        "--silence-search-window-seconds",
        str(args.silence_search_window_seconds),
        "--silence-noise-db",
        str(args.silence_noise_db),
        "--silence-min-duration",
        str(args.silence_min_duration),
        "--dynamic-lexicon-terms",
        str(args.dynamic_lexicon_terms),
        "--dynamic-lexicon-min-len",
        str(args.dynamic_lexicon_min_len),
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
        "--quality-threshold",
        str(args.quality_threshold),
        "--run-throughput",
        "--throughput-scenario",
        scenario_name,
        "--throughput-workers",
        *[str(w) for w in throughput_workers],
    ]
    if gold_artifact_dir:
        cmd.extend(["--gold-artifact-dir", str(gold_artifact_dir)])
    if skip_gold:
        cmd.append("--skip-gold")

    artifact_dir = _run_command_capture(cmd, log_path)
    return Path(artifact_dir).resolve()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _quality_metrics(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    target = [
        row
        for row in rows
        if str(row.get("scenario")) == scenario
        and str(row.get("file_id")) != "__aggregate__"
        and _safe_float(row.get("word_drift")) is not None
    ]
    word_vals = [_safe_float(r.get("word_drift")) for r in target]
    seam_vals = [_safe_float(r.get("seam_word_drift")) for r in target]
    char_vals = [_safe_float(r.get("char_drift")) for r in target]
    rtf_vals = [_safe_float(r.get("rtf")) for r in target]
    return {
        "quality_runs": len(target),
        "mean_word_drift": _mean([v for v in word_vals if v is not None]),
        "mean_seam_word_drift": _mean([v for v in seam_vals if v is not None]),
        "mean_char_drift": _mean([v for v in char_vals if v is not None]),
        "mean_rtf": _mean([v for v in rtf_vals if v is not None]),
    }


def _throughput_aggregates(rows: list[dict[str, Any]], scenario: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("file_id")) != "__aggregate__":
            continue
        name = str(row.get("scenario", ""))
        match = None
        if name.startswith("throughput_workers_") and name.endswith(f"_{scenario}"):
            middle = name[len("throughput_workers_") : -len(f"_{scenario}")]
            try:
                match = int(middle)
            except ValueError:
                match = None
        if match is None:
            continue
        out[match] = row
    return out


def _max_stable_worker(
    throughput_by_worker: dict[int, dict[str, Any]],
    worker_grid: list[int],
) -> int:
    stable = []
    for worker in worker_grid:
        row = throughput_by_worker.get(worker)
        if not row:
            continue
        success = _parse_bool(row.get("success"))
        chunk_failures = _safe_int(row.get("chunk_failures")) or 0
        if success and chunk_failures == 0:
            stable.append(worker)
    if not stable:
        return 1
    return max(stable)


def _throughput_drop_by_worker(
    baseline_tp: dict[int, dict[str, Any]],
    candidate_tp: dict[int, dict[str, Any]],
) -> dict[int, float]:
    drops: dict[int, float] = {}
    for worker, base_row in baseline_tp.items():
        cand_row = candidate_tp.get(worker)
        if not cand_row:
            continue
        base_val = _safe_float(base_row.get("throughput_audio_hours_per_hour"))
        cand_val = _safe_float(cand_row.get("throughput_audio_hours_per_hour"))
        if base_val is None or cand_val is None or base_val <= 0:
            continue
        drops[worker] = max(0.0, (base_val - cand_val) / base_val)
    return drops


def _build_score_row(
    label: str,
    variant_spec: str,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    scenario: str,
    preferred_worker: int,
    perf_budget_drop: float,
    phase_name: str,
) -> dict[str, Any]:
    quality = _quality_metrics(rows, scenario=scenario)
    base_quality = _quality_metrics(baseline_rows, scenario=scenario)
    tp = _throughput_aggregates(rows, scenario=scenario)
    base_tp = _throughput_aggregates(baseline_rows, scenario=scenario)
    drop_map = _throughput_drop_by_worker(base_tp, tp)

    base_word = _safe_float(base_quality.get("mean_word_drift"))
    cand_word = _safe_float(quality.get("mean_word_drift"))
    quality_improved = bool(
        base_word is not None and cand_word is not None and cand_word < base_word
    )
    preferred_drop = drop_map.get(preferred_worker)
    pass_perf = preferred_drop is not None and preferred_drop <= perf_budget_drop

    preferred_tp_audio = None
    preferred_tp_files = None
    if preferred_worker in tp:
        preferred_tp_audio = _safe_float(tp[preferred_worker].get("throughput_audio_hours_per_hour"))
        preferred_tp_files = _safe_float(tp[preferred_worker].get("throughput_files_per_hour"))

    return {
        "phase": phase_name,
        "label": label,
        "variant_spec": variant_spec,
        "scenario": scenario,
        "preferred_worker": preferred_worker,
        "quality_runs": quality["quality_runs"],
        "mean_word_drift": quality["mean_word_drift"],
        "mean_seam_word_drift": quality["mean_seam_word_drift"],
        "mean_char_drift": quality["mean_char_drift"],
        "mean_rtf": quality["mean_rtf"],
        "preferred_throughput_audio_hours_per_hour": preferred_tp_audio,
        "preferred_throughput_files_per_hour": preferred_tp_files,
        "preferred_throughput_drop_pct_vs_baseline": (
            None if preferred_drop is None else round(preferred_drop * 100.0, 6)
        ),
        "throughput_drop_pct_by_worker_json": json.dumps(
            {str(k): round(v * 100.0, 6) for k, v in sorted(drop_map.items())},
            ensure_ascii=True,
            sort_keys=True,
        ),
        "quality_improved_vs_baseline": quality_improved,
        "pass_perf_budget": pass_perf,
        "eligible": bool(quality_improved and pass_perf),
        "generated_at": _utc_now_iso(),
    }


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


def _write_scoreboard_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.append("| Label | Mean Word Drift | Mean Seam Drift | Worker | Audio Hr/Hr | Drop % | Eligible |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {label} | {word} | {seam} | {worker} | {ahph} | {drop} | {eligible} |".format(
                label=row.get("label"),
                word=(
                    f"{float(row['mean_word_drift']):.4f}"
                    if row.get("mean_word_drift") is not None
                    else "NA"
                ),
                seam=(
                    f"{float(row['mean_seam_word_drift']):.4f}"
                    if row.get("mean_seam_word_drift") is not None
                    else "NA"
                ),
                worker=row.get("preferred_worker"),
                ahph=(
                    f"{float(row['preferred_throughput_audio_hours_per_hour']):.3f}"
                    if row.get("preferred_throughput_audio_hours_per_hour") is not None
                    else "NA"
                ),
                drop=(
                    f"{float(row['preferred_throughput_drop_pct_vs_baseline']):.2f}"
                    if row.get("preferred_throughput_drop_pct_vs_baseline") is not None
                    else "NA"
                ),
                eligible=row.get("eligible"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rank_best(
    rows: list[dict[str, Any]],
    require_eligible: bool,
) -> dict[str, Any] | None:
    pool = [row for row in rows if bool(row.get("eligible"))] if require_eligible else list(rows)
    if not pool:
        return None

    def _key(row: dict[str, Any]) -> tuple[float, float]:
        drift = _safe_float(row.get("mean_word_drift"))
        drop = _safe_float(row.get("preferred_throughput_drop_pct_vs_baseline"))
        drift_key = drift if drift is not None else float("inf")
        drop_key = drop if drop is not None else float("inf")
        return (drift_key, drop_key)

    return sorted(pool, key=_key)[0]


def _variant_union_spec(*aliases: str) -> str:
    features: set[str] = set()
    for alias in aliases:
        features.update(VARIANT_ALIASES[alias])
    if not features:
        return "baseline_overlap_context"
    return ",".join(sorted(features))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two-phase idea matrix experiments for VibeVoice ASR chunking."
    )
    parser.add_argument(
        "--phase1-manifest",
        default="vllm_plugin/experiments/manifests/endpoint_02172026_phase1_8.json",
        help="Phase-1 (pilot) manifest path.",
    )
    parser.add_argument(
        "--phase2-manifest",
        default="vllm_plugin/experiments/manifests/endpoint_02172026_phase2_23.json",
        help="Phase-2 (full) manifest path.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/chunking_experiment/idea_matrix",
        help="Output root for matrix artifacts.",
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="vLLM OpenAI-compatible endpoint URL.",
    )
    parser.add_argument(
        "--model",
        default="vibevoice",
        help="Served model name.",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=int,
        default=30,
        help="Single chunk size used for the matrix.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=30.0,
        help="Chunk overlap seconds.",
    )
    parser.add_argument(
        "--context-tail-chars",
        type=int,
        default=800,
        help="Context tail carry length.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.02,
        help="Word drift pass threshold.",
    )
    parser.add_argument(
        "--perf-budget-drop",
        type=float,
        default=0.20,
        help="Maximum allowed throughput drop ratio vs baseline.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum worker count in the sweep grid.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="max_tokens for requests.",
    )
    parser.add_argument(
        "--max-tokens-gold",
        type=int,
        default=32768,
        help="max_tokens for gold requests.",
    )
    parser.add_argument(
        "--max-tokens-chunk",
        type=int,
        default=24576,
        help="max_tokens for chunk requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Request retry count.",
    )
    parser.add_argument(
        "--seam-window-seconds",
        type=float,
        default=60.0,
        help="Seam metric/selection window.",
    )
    parser.add_argument(
        "--micro-redo-window-seconds",
        type=float,
        default=90.0,
        help="Micro-redo half window.",
    )
    parser.add_argument(
        "--shifted-offset-ratio",
        type=float,
        default=0.5,
        help="Shifted-grid offset ratio of chunk size.",
    )
    parser.add_argument(
        "--silence-search-window-seconds",
        type=float,
        default=45.0,
        help="Boundary silence alignment search window.",
    )
    parser.add_argument(
        "--silence-noise-db",
        type=float,
        default=-35.0,
        help="silencedetect threshold in dB.",
    )
    parser.add_argument(
        "--silence-min-duration",
        type=float,
        default=0.3,
        help="silencedetect minimum silence duration.",
    )
    parser.add_argument(
        "--dynamic-lexicon-terms",
        type=int,
        default=20,
        help="Max dynamic hotword terms.",
    )
    parser.add_argument(
        "--dynamic-lexicon-min-len",
        type=int,
        default=4,
        help="Dynamic lexicon min term length.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario = _scenario_name(args.chunk_minutes)
    workers = _worker_grid(args.max_workers)
    if not workers:
        raise ValueError("No workers selected. Increase --max-workers.")

    phase1_manifest = Path(args.phase1_manifest).resolve()
    phase2_manifest = Path(args.phase2_manifest).resolve()
    if not phase1_manifest.exists():
        raise FileNotFoundError(f"Phase1 manifest missing: {phase1_manifest}")
    if not phase2_manifest.exists():
        raise FileNotFoundError(f"Phase2 manifest missing: {phase2_manifest}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.output_dir).resolve() / timestamp
    phase1_root = root / "phase1"
    phase2_root = root / "phase2"
    phase1_root.mkdir(parents=True, exist_ok=True)
    phase2_root.mkdir(parents=True, exist_ok=True)

    print("[INFO] Phase 1 baseline run (worker sweep).")
    p1_baseline_artifact = _run_variant_experiment(
        phase_root=phase1_root,
        label="baseline",
        variant_spec="baseline_overlap_context",
        manifest_path=phase1_manifest,
        args=args,
        throughput_workers=workers,
        scenario_name=scenario,
        gold_artifact_dir=None,
        skip_gold=False,
    )
    p1_baseline_rows = _read_summary_rows(p1_baseline_artifact / "summary.csv")
    p1_baseline_tp = _throughput_aggregates(p1_baseline_rows, scenario=scenario)
    p1_max_stable = _max_stable_worker(p1_baseline_tp, worker_grid=workers)
    p1_test_workers = sorted({1, p1_max_stable})

    print(f"[INFO] Phase 1 max stable worker: {p1_max_stable}")
    phase1_variant_rows: list[dict[str, Any]] = []
    phase1_variant_runs: dict[str, tuple[str, Path, list[dict[str, Any]]]] = {}

    baseline_score = _build_score_row(
        label="baseline",
        variant_spec="baseline_overlap_context",
        rows=p1_baseline_rows,
        baseline_rows=p1_baseline_rows,
        scenario=scenario,
        preferred_worker=p1_max_stable,
        perf_budget_drop=args.perf_budget_drop,
        phase_name="phase1",
    )
    phase1_variant_rows.append(baseline_score)
    phase1_variant_runs["baseline"] = ("baseline_overlap_context", p1_baseline_artifact, p1_baseline_rows)

    for alias in IDEA_ORDER:
        print(f"[INFO] Phase 1 single idea run: {alias}")
        artifact = _run_variant_experiment(
            phase_root=phase1_root,
            label=alias,
            variant_spec=alias,
            manifest_path=phase1_manifest,
            args=args,
            throughput_workers=p1_test_workers,
            scenario_name=scenario,
            gold_artifact_dir=p1_baseline_artifact,
            skip_gold=True,
        )
        rows = _read_summary_rows(artifact / "summary.csv")
        score = _build_score_row(
            label=alias,
            variant_spec=alias,
            rows=rows,
            baseline_rows=p1_baseline_rows,
            scenario=scenario,
            preferred_worker=p1_max_stable,
            perf_budget_drop=args.perf_budget_drop,
            phase_name="phase1",
        )
        phase1_variant_rows.append(score)
        phase1_variant_runs[alias] = (alias, artifact, rows)

    best_single = _rank_best(
        [row for row in phase1_variant_rows if row["label"] != "baseline"],
        require_eligible=True,
    ) or _rank_best(
        [row for row in phase1_variant_rows if row["label"] != "baseline"],
        require_eligible=False,
    )
    if not best_single:
        raise RuntimeError("Failed to choose a best single idea.")
    best_single_alias = str(best_single["label"])
    print(f"[INFO] Phase 1 best single: {best_single_alias}")

    phase1_combo_rows: list[dict[str, Any]] = []
    pair_labels: list[str] = []
    for alias in IDEA_ORDER:
        if alias == best_single_alias:
            continue
        label = f"pair__{best_single_alias}__{alias}"
        pair_labels.append(label)
        variant_spec = _variant_union_spec(best_single_alias, alias)
        print(f"[INFO] Phase 1 pair combo run: {label} -> {variant_spec}")
        artifact = _run_variant_experiment(
            phase_root=phase1_root,
            label=label,
            variant_spec=variant_spec,
            manifest_path=phase1_manifest,
            args=args,
            throughput_workers=p1_test_workers,
            scenario_name=scenario,
            gold_artifact_dir=p1_baseline_artifact,
            skip_gold=True,
        )
        rows = _read_summary_rows(artifact / "summary.csv")
        score = _build_score_row(
            label=label,
            variant_spec=variant_spec,
            rows=rows,
            baseline_rows=p1_baseline_rows,
            scenario=scenario,
            preferred_worker=p1_max_stable,
            perf_budget_drop=args.perf_budget_drop,
            phase_name="phase1",
        )
        phase1_combo_rows.append(score)
        phase1_variant_runs[label] = (variant_spec, artifact, rows)

    best_pair = _rank_best(phase1_combo_rows, require_eligible=True)
    best_pair_row = None
    if best_pair:
        best_single_word = _safe_float(best_single.get("mean_word_drift"))
        best_pair_word = _safe_float(best_pair.get("mean_word_drift"))
        if (
            best_single_word is not None
            and best_pair_word is not None
            and best_pair_word < best_single_word
        ):
            best_pair_row = best_pair

    best_triple_row = None
    if best_pair_row:
        pair_label = str(best_pair_row["label"])
        _, a, b = pair_label.split("__", 2)
        used = {a, b}
        remaining = [alias for alias in IDEA_ORDER if alias not in used]
        if remaining:
            remaining_rows = [
                row
                for row in phase1_variant_rows
                if row["label"] in remaining
            ]
            best_remaining = _rank_best(remaining_rows, require_eligible=False)
            if best_remaining:
                remaining_alias = str(best_remaining["label"])
                triple_label = f"triple__{a}__{b}__{remaining_alias}"
                variant_spec = _variant_union_spec(a, b, remaining_alias)
                print(f"[INFO] Phase 1 triple combo run: {triple_label} -> {variant_spec}")
                artifact = _run_variant_experiment(
                    phase_root=phase1_root,
                    label=triple_label,
                    variant_spec=variant_spec,
                    manifest_path=phase1_manifest,
                    args=args,
                    throughput_workers=p1_test_workers,
                    scenario_name=scenario,
                    gold_artifact_dir=p1_baseline_artifact,
                    skip_gold=True,
                )
                rows = _read_summary_rows(artifact / "summary.csv")
                score = _build_score_row(
                    label=triple_label,
                    variant_spec=variant_spec,
                    rows=rows,
                    baseline_rows=p1_baseline_rows,
                    scenario=scenario,
                    preferred_worker=p1_max_stable,
                    perf_budget_drop=args.perf_budget_drop,
                    phase_name="phase1",
                )
                phase1_combo_rows.append(score)
                phase1_variant_runs[triple_label] = (variant_spec, artifact, rows)
                if _parse_bool(score.get("eligible")):
                    pair_word = _safe_float(best_pair_row.get("mean_word_drift"))
                    triple_word = _safe_float(score.get("mean_word_drift"))
                    if (
                        pair_word is not None
                        and triple_word is not None
                        and triple_word < pair_word
                    ):
                        best_triple_row = score

    finalists: list[tuple[str, str]] = [("baseline", "baseline_overlap_context")]
    finalists.append((best_single_alias, phase1_variant_runs[best_single_alias][0]))
    if best_pair_row:
        pair_label = str(best_pair_row["label"])
        finalists.append((pair_label, phase1_variant_runs[pair_label][0]))
    if best_triple_row:
        triple_label = str(best_triple_row["label"])
        finalists.append((triple_label, phase1_variant_runs[triple_label][0]))
    # de-duplicate by label preserving order
    deduped_finalists: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for label, variant_spec in finalists:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        deduped_finalists.append((label, variant_spec))
    finalists = deduped_finalists

    print("[INFO] Phase 2 baseline run (worker sweep).")
    p2_baseline_artifact = _run_variant_experiment(
        phase_root=phase2_root,
        label="baseline",
        variant_spec="baseline_overlap_context",
        manifest_path=phase2_manifest,
        args=args,
        throughput_workers=workers,
        scenario_name=scenario,
        gold_artifact_dir=None,
        skip_gold=False,
    )
    p2_baseline_rows = _read_summary_rows(p2_baseline_artifact / "summary.csv")
    p2_baseline_tp = _throughput_aggregates(p2_baseline_rows, scenario=scenario)
    p2_max_stable = _max_stable_worker(p2_baseline_tp, worker_grid=workers)
    p2_workers = [w for w in workers if w <= p2_max_stable]
    if not p2_workers:
        p2_workers = [1]
    print(f"[INFO] Phase 2 max stable worker: {p2_max_stable}")

    phase2_rows: list[dict[str, Any]] = []
    p2_baseline_score = _build_score_row(
        label="baseline",
        variant_spec="baseline_overlap_context",
        rows=p2_baseline_rows,
        baseline_rows=p2_baseline_rows,
        scenario=scenario,
        preferred_worker=p2_max_stable,
        perf_budget_drop=args.perf_budget_drop,
        phase_name="phase2",
    )
    phase2_rows.append(p2_baseline_score)

    for label, variant_spec in finalists:
        if label == "baseline":
            continue
        print(f"[INFO] Phase 2 finalist run: {label} -> {variant_spec}")
        artifact = _run_variant_experiment(
            phase_root=phase2_root,
            label=label,
            variant_spec=variant_spec,
            manifest_path=phase2_manifest,
            args=args,
            throughput_workers=p2_workers,
            scenario_name=scenario,
            gold_artifact_dir=p2_baseline_artifact,
            skip_gold=True,
        )
        rows = _read_summary_rows(artifact / "summary.csv")
        score = _build_score_row(
            label=label,
            variant_spec=variant_spec,
            rows=rows,
            baseline_rows=p2_baseline_rows,
            scenario=scenario,
            preferred_worker=p2_max_stable,
            perf_budget_drop=args.perf_budget_drop,
            phase_name="phase2",
        )
        phase2_rows.append(score)

    phase1_variant_csv = phase1_root / "variant_scoreboard.csv"
    phase1_variant_md = phase1_root / "variant_scoreboard.md"
    phase1_combo_csv = phase1_root / "combo_scoreboard.csv"
    phase1_combo_md = phase1_root / "combo_scoreboard.md"
    phase2_csv = phase2_root / "variant_scoreboard.csv"
    phase2_md = phase2_root / "variant_scoreboard.md"

    _write_csv(phase1_variant_csv, phase1_variant_rows)
    _write_scoreboard_md(phase1_variant_md, "Phase1 Variant Scoreboard", phase1_variant_rows)
    _write_csv(phase1_combo_csv, phase1_combo_rows)
    _write_scoreboard_md(phase1_combo_md, "Phase1 Combo Scoreboard", phase1_combo_rows)
    _write_csv(phase2_csv, phase2_rows)
    _write_scoreboard_md(phase2_md, "Phase2 Variant Scoreboard", phase2_rows)

    eligible_phase2 = [row for row in phase2_rows if row["label"] != "baseline" and _parse_bool(row.get("eligible"))]
    winner = _rank_best(eligible_phase2, require_eligible=False)
    if not winner:
        winner = p2_baseline_score

    recommendation = {
        "generated_at": _utc_now_iso(),
        "api_url": args.api_url,
        "scenario": scenario,
        "phase1_max_stable_worker": p1_max_stable,
        "phase2_max_stable_worker": p2_max_stable,
        "perf_budget_drop_ratio": args.perf_budget_drop,
        "finalists": [label for label, _ in finalists],
        "winner": winner,
        "paths": {
            "phase1_variant_scoreboard_csv": str(phase1_variant_csv),
            "phase1_combo_scoreboard_csv": str(phase1_combo_csv),
            "phase2_variant_scoreboard_csv": str(phase2_csv),
        },
    }
    recommendation_path = root / "final_recommendation.json"
    recommendation_path.write_text(
        json.dumps(recommendation, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print("[INFO] Idea matrix complete.")
    print(f"[INFO] Root output: {root}")
    print(f"[INFO] Final recommendation: {recommendation_path}")
    print(json.dumps(winner, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
