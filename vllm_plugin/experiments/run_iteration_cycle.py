#!/usr/bin/env python3
"""
Manage continuous ASR iteration cycles.

This script adds a thin orchestration layer on top of the existing experiment
stack so iteration can follow a repeatable loop:

1. Prepare a cycle from the current program manifest and backlog
2. Execute the validation ladder for runnable challengers
3. Finalize the cycle after expert review has been scored

The expert transcript rubric remains the primary quality signal. Metric
scoreboards are supporting signals used for throughput and tie-breaks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from vllm_plugin.experiments.expert_review import summarize_review_payload, write_review_summary
except ModuleNotFoundError:
    from expert_review import summarize_review_payload, write_review_summary


VERDICT_RANK = {
    "not_acceptable": 0,
    "borderline": 1,
    "acceptable": 2,
    "equivalent": 3,
}
TERMINAL_STATUSES = {"promoted", "completed", "killed"}
DEFAULT_PROGRAM_MANIFEST = "vllm_plugin/experiments/manifests/asr_iteration_program.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scenario_name(chunk_minutes: int) -> str:
    return f"chunk_{chunk_minutes}m_overlap_context"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def compute_priority_score(item: dict[str, Any]) -> int:
    return (
        3 * int(item.get("impact", 0))
        + 2 * int(item.get("failure_severity", 0))
        + int(item.get("confidence", 0))
        + int(item.get("reusability", 0))
        - int(item.get("effort", 0))
        - int(item.get("runtime_cost", 0))
    )


def select_cycle_candidates(backlog: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = []
    for index, item in enumerate(backlog):
        clone = dict(item)
        clone["priority_score"] = compute_priority_score(item)
        clone["_source_index"] = index
        ranked.append(clone)
    ranked.sort(
        key=lambda item: (
            -item["priority_score"],
            -int(item.get("impact", 0)),
            -int(item.get("failure_severity", 0)),
            int(item.get("effort", 0)),
            int(item.get("_source_index", 0)),
        )
    )

    candidates = [item for item in ranked if item.get("status") not in TERMINAL_STATUSES]
    selected: list[dict[str, Any]] = []
    covered_failure_modes: set[str] = set()

    while candidates and len(selected) < limit:
        chosen_idx = 0
        if len(selected) == 1:
            for index, item in enumerate(candidates):
                item_modes = set(item.get("target_failure_modes", []))
                if item_modes - covered_failure_modes:
                    chosen_idx = index
                    break
        chosen = candidates.pop(chosen_idx)
        covered_failure_modes.update(chosen.get("target_failure_modes", []))
        selected.append(chosen)

    for row in ranked:
        row.pop("_source_index", None)
    for row in selected:
        row.pop("_source_index", None)
    return selected, ranked


def _validate_hook_policy(program: dict[str, Any]) -> None:
    hook_policy = dict(program.get("hook_policy", {}))
    required_fields = [str(value) for value in hook_policy.get("required_backlog_fields", [])]
    if not required_fields:
        return
    enforced_statuses = {"active", "ready", "promoted"}
    issues: list[str] = []
    for item in program.get("backlog", []):
        status = str(item.get("status", "planned"))
        if status not in enforced_statuses:
            continue
        item_id = str(item.get("id") or "<missing-id>")
        for field in required_fields:
            value = item.get(field)
            if value in (None, ""):
                issues.append(f"{item_id}: missing required field '{field}'")
    if issues:
        raise ValueError(
            "Backlog hook policy validation failed:\n- " + "\n- ".join(issues)
        )


def _verdict_rank(verdict: str | None) -> int:
    return VERDICT_RANK.get(str(verdict or "not_acceptable"), 0)


def _verdict_meets(verdict: str | None, minimum: str) -> bool:
    return _verdict_rank(verdict) >= _verdict_rank(minimum)


def _stage_enabled(dataset_cfg: dict[str, Any]) -> bool:
    return _parse_bool(dataset_cfg.get("enabled", True))


def _group_stage_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("variant")), str(row.get("scenario")))] = row
    return out


def _item_stage_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        out[(str(row.get("variant")), str(row.get("scenario")), str(row.get("file_id")))] = row
    return out


def _review_payload_has_scores(payload: dict[str, Any]) -> bool:
    for item in payload.get("items", []):
        review = dict(item.get("review", {}))
        if _parse_bool(review.get("exclude_from_ranking")):
            return True
        scores = review.get("scores", {})
        if any(value not in (None, "") for value in dict(scores).values()):
            return True
    return False


def _ensure_review_summary(stage_run_root: Path) -> dict[str, Path] | None:
    review_dir = stage_run_root / "expert_review"
    review_json = review_dir / "review_template.json"
    if not review_json.exists():
        return None
    scoreboard_csv = review_dir / "scoreboard.csv"
    item_scores_csv = review_dir / "item_scores.csv"
    if scoreboard_csv.exists() and item_scores_csv.exists():
        return {
            "scoreboard_csv": scoreboard_csv,
            "item_scores_csv": item_scores_csv,
        }
    payload = _read_json(review_json)
    if not _review_payload_has_scores(payload):
        return None
    summary = summarize_review_payload(payload)
    outputs = write_review_summary(review_dir, summary)
    return outputs


def _render_program_note(
    program: dict[str, Any],
    ranked_backlog: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
) -> str:
    champion = dict(program.get("champion", {}))
    current_findings = dict(program.get("current_findings", {}))
    lines = [
        "# ASR Iteration Program",
        "",
        f"- Updated: `{_utc_now_iso()}`",
        f"- Program id: `{program.get('program_id')}`",
        f"- Current champion: `{champion.get('label')}` ({champion.get('expert_score')} / `{champion.get('verdict')}`)",
        f"- Active cycle: `{program.get('active_cycle_id') or 'none'}`",
        f"- Consecutive production passes: `{program.get('consecutive_production_passes', 0)}` / `{program.get('production', {}).get('required_consecutive_passes', 2)}`",
        "",
        "## Current Findings",
        "",
    ]
    for failure_mode in current_findings.get("failure_modes", []):
        lines.append(f"- `{failure_mode}`")
    lines.extend(
        [
            "",
            "## Sentinel Files",
            "",
        ]
    )
    for file_id in current_findings.get("sentinel_file_ids", []):
        lines.append(f"- `{file_id}`")
    lines.extend(
        [
            "",
            "## Next Queue",
            "",
        ]
    )
    for item in selected_candidates:
        lines.append(
            "- `{id}` priority `{score}` status `{status}` targets `{targets}`".format(
                id=item.get("id"),
                score=item.get("priority_score"),
                status=item.get("status"),
                targets=", ".join(item.get("target_failure_modes", [])) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Artifact root: `{current_findings.get('artifact_root')}`",
            f"- Expert review scoreboard: `{current_findings.get('expert_review_scoreboard')}`",
            f"- Item scores: `{current_findings.get('item_scores_csv')}`",
        ]
    )
    return "\n".join(lines)


def _render_backlog_note(ranked_backlog: list[dict[str, Any]]) -> str:
    lines = [
        "# ASR Hypothesis Backlog",
        "",
        "| Idea | Status | Priority | Runnable | Targets | Hypothesis |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in ranked_backlog:
        lines.append(
            "| {idea} | {status} | {priority} | {runnable} | {targets} | {hypothesis} |".format(
                idea=item.get("id"),
                status=item.get("status"),
                priority=item.get("priority_score"),
                runnable=bool(item.get("runnable_variant")),
                targets=", ".join(item.get("target_failure_modes", [])) or "none",
                hypothesis=str(item.get("hypothesis", "")).replace("|", "/"),
            )
        )
    return "\n".join(lines)


def _render_cycle_note(cycle_spec: dict[str, Any], decision_summary: dict[str, Any] | None = None) -> str:
    non_runnable = [item["id"] for item in cycle_spec.get("selected_candidates", []) if not item.get("runnable_variant")]
    lines = [
        f"# ASR Iteration {cycle_spec['cycle_id']}",
        "",
        f"- Created: `{cycle_spec.get('generated_at')}`",
        f"- Status: `{decision_summary.get('status') if decision_summary else cycle_spec.get('status', 'planned')}`",
        f"- Champion: `{cycle_spec['champion']['label']}`",
        "",
        "## Selected Ideas",
        "",
    ]
    for item in cycle_spec.get("selected_candidates", []):
        lines.append(
            "- `{id}` priority `{score}` targets `{targets}` runnable `{runnable}`".format(
                id=item.get("id"),
                score=item.get("priority_score"),
                targets=", ".join(item.get("target_failure_modes", [])) or "none",
                runnable=bool(item.get("runnable_variant")),
            )
        )
        lines.append(f"  hypothesis: {item.get('hypothesis')}")
    lines.extend(
        [
            "",
            "## Validation Ladder",
            "",
        ]
    )
    for stage_name, stage_cfg in cycle_spec.get("datasets", {}).items():
        if not _stage_enabled(stage_cfg):
            continue
        lines.append(
            "- `{name}` manifest `{manifest}` chunk_minutes `{chunks}` throughput `{throughput}`".format(
                name=stage_name,
                manifest=stage_cfg.get("manifest"),
                chunks=stage_cfg.get("chunk_minutes"),
                throughput=bool(stage_cfg.get("run_throughput")),
            )
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
        ]
    )
    for gate_name, gate_cfg in cycle_spec.get("gates", {}).items():
        lines.append(f"- `{gate_name}`: `{json.dumps(gate_cfg, ensure_ascii=True, sort_keys=True)}`")
    if non_runnable:
        lines.extend(
            [
                "",
                "## Execution Status",
                "",
                "- Blocked on implementation wiring for: `{}`".format("`, `".join(non_runnable)),
                "- Update `runnable_variant` in `vllm_plugin/experiments/manifests/asr_iteration_program.json` once each idea has a harness entrypoint.",
            ]
        )
    if decision_summary:
        lines.extend(
            [
                "",
                "## Results",
                "",
                f"- Promotion: `{decision_summary.get('promoted_candidate') or 'none'}`",
                f"- Production ready: `{decision_summary.get('production_ready')}`",
                "",
            ]
        )
        for candidate in decision_summary.get("candidate_results", []):
            lines.append(
                "- `{id}` overall `{overall}` smoke `{smoke}` sentinel `{sentinel}` long_tail `{long_tail}` throughput `{throughput}`".format(
                    id=candidate.get("id"),
                    overall=candidate.get("overall_recommendation"),
                    smoke=candidate.get("smoke", {}).get("status"),
                    sentinel=candidate.get("sentinel", {}).get("status"),
                    long_tail=candidate.get("long_tail", {}).get("status"),
                    throughput=candidate.get("throughput", {}).get("status"),
                )
            )
    return "\n".join(lines)


def _build_cycle_spec(program: dict[str, Any], cycle_id: str, selected_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = Path(program["output_dir"]).resolve()
    cycle_root = output_dir / cycle_id
    notes_dir = Path(program["notes_dir"]).resolve()
    defaults = dict(program.get("defaults", {}))
    return {
        "cycle_id": cycle_id,
        "generated_at": _utc_now_iso(),
        "status": "planned",
        "program_manifest": str(Path(program["_program_manifest"]).resolve()),
        "cycle_root": str(cycle_root),
        "notes_dir": str(notes_dir),
        "notes_paths": {
            "program": str(notes_dir / "asr_iteration_program.md"),
            "backlog": str(notes_dir / "asr_hypothesis_backlog.md"),
            "cycle": str(notes_dir / f"asr_iteration_{cycle_id}.md"),
        },
        "api_url": program["api_url"],
        "model": program.get("model", "vibevoice"),
        "champion": dict(program["champion"]),
        "selected_candidates": selected_candidates,
        "datasets": dict(program.get("datasets", {})),
        "gates": dict(program.get("gates", {})),
        "defaults": defaults,
    }


def _run_command_capture(cmd: list[str], log_path: Path, root_prefix: str) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    found_root = ""
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
            if root_prefix in line:
                found_root = line.split(root_prefix, 1)[1].strip()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")
    if not found_root:
        raise RuntimeError(f"Could not parse run root from log: {log_path}")
    return found_root


def _candidate_variants(cycle_spec: dict[str, Any]) -> list[str]:
    variants = ["baseline_overlap_context", str(cycle_spec["champion"]["variant"])]
    for candidate in cycle_spec.get("selected_candidates", []):
        variant = candidate.get("runnable_variant")
        if variant:
            variants.append(str(variant))
    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _run_stage(cycle_spec: dict[str, Any], stage_name: str) -> Path:
    dataset_cfg = dict(cycle_spec["datasets"][stage_name])
    stage_base = Path(cycle_spec["cycle_root"]) / "stages" / stage_name
    defaults = dict(cycle_spec.get("defaults", {}))
    variants = _candidate_variants(cycle_spec)
    cmd = [
        "python3",
        "-u",
        "vllm_plugin/experiments/run_finalist_validation.py",
        "--api-url",
        cycle_spec["api_url"],
        "--manifest",
        str(Path(dataset_cfg["manifest"]).resolve()),
        "--output-dir",
        str(stage_base),
        "--model",
        cycle_spec["model"],
        "--variants",
        *variants,
        "--chunk-minutes",
        *[str(value) for value in dataset_cfg.get("chunk_minutes", [30])],
        "--overlap-seconds",
        str(dataset_cfg.get("overlap_seconds", defaults.get("overlap_seconds", 30))),
        "--context-tail-chars",
        str(dataset_cfg.get("context_tail_chars", defaults.get("context_tail_chars", 800))),
        "--quality-threshold",
        str(dataset_cfg.get("quality_threshold", defaults.get("quality_threshold", 0.02))),
        "--max-tokens",
        str(dataset_cfg.get("max_tokens", defaults.get("max_tokens", 32768))),
        "--max-tokens-gold",
        str(dataset_cfg.get("max_tokens_gold", defaults.get("max_tokens_gold", 32768))),
        "--max-tokens-chunk",
        str(dataset_cfg.get("max_tokens_chunk", defaults.get("max_tokens_chunk", 24576))),
        "--timeout-seconds",
        str(dataset_cfg.get("timeout_seconds", defaults.get("timeout_seconds", 7200))),
        "--max-retries",
        str(dataset_cfg.get("max_retries", defaults.get("max_retries", 1))),
    ]
    if _parse_bool(dataset_cfg.get("run_throughput")):
        cmd.extend(
            [
                "--run-throughput",
                "--throughput-workers",
                *[str(value) for value in dataset_cfg.get("throughput_workers", defaults.get("throughput_workers", [1, 2, 4]))],
            ]
        )
    if dataset_cfg.get("baseline_gold_artifact_dir"):
        cmd.extend(
            [
                "--baseline-gold-artifact-dir",
                str(Path(str(dataset_cfg["baseline_gold_artifact_dir"])).resolve()),
            ]
        )
    if _parse_bool(dataset_cfg.get("baseline_skip_gold")):
        cmd.append("--baseline-skip-gold")
    log_path = Path(cycle_spec["cycle_root"]) / "logs" / f"{stage_name}.log"
    stage_run_root = _run_command_capture(cmd, log_path, "Root output:")
    return Path(stage_run_root).resolve()


def _find_variant_artifact_dir(stage_run_root: Path, variant: str) -> Path | None:
    variant_root = stage_run_root / "runs" / variant
    if not variant_root.exists():
        return None
    dirs = sorted([path for path in variant_root.iterdir() if path.is_dir()])
    if not dirs:
        return None
    return dirs[-1]


def _throughput_rows(summary_rows: list[dict[str, Any]], scenario: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    suffix = f"_{scenario}"
    for row in summary_rows:
        if str(row.get("file_id")) != "__aggregate__":
            continue
        name = str(row.get("scenario", ""))
        if not name.startswith("throughput_workers_") or not name.endswith(suffix):
            continue
        middle = name[len("throughput_workers_") : -len(suffix)]
        worker = _safe_int(middle)
        if worker is None:
            continue
        rows[worker] = row
    return rows


def _evaluate_smoke_gate(
    candidate_id: str,
    candidate_variant: str,
    *,
    champion_variant: str,
    scenario: str,
    grouped_rows: dict[tuple[str, str], dict[str, Any]],
    item_rows: dict[tuple[str, str, str], dict[str, Any]],
    gate_cfg: dict[str, Any],
) -> dict[str, Any]:
    candidate = grouped_rows.get((candidate_variant, scenario))
    champion = grouped_rows.get((champion_variant, scenario))
    if not candidate or not champion:
        return {"status": "pending_review", "reason": "Missing stage review summary."}
    candidate_score = _safe_float(candidate.get("mean_weighted_score"))
    champion_score = _safe_float(champion.get("mean_weighted_score"))
    delta = None if candidate_score is None or champion_score is None else round(candidate_score - champion_score, 2)
    meaning_loss = any(
        _parse_bool(row.get("meaning_loss"))
        for key, row in item_rows.items()
        if key[0] == candidate_variant and key[1] == scenario
    )
    candidate_verdict = str(candidate.get("overall_verdict"))
    champion_verdict = str(champion.get("overall_verdict"))
    pass_verdict = candidate_verdict != "not_acceptable"
    pass_critical = (_safe_int(candidate.get("critical_issues")) or 0) == 0
    pass_meaning = not meaning_loss
    score_ok = delta is not None and delta >= float(gate_cfg.get("min_score_delta", 5.0))
    verdict_improved = _verdict_rank(candidate_verdict) > _verdict_rank(champion_verdict)
    passed = bool(pass_verdict and pass_critical and pass_meaning and (score_ok or verdict_improved))
    return {
        "status": "pass" if passed else "fail",
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "score_delta": delta,
        "candidate_verdict": candidate_verdict,
        "champion_verdict": champion_verdict,
        "meaning_loss": meaning_loss,
    }


def _evaluate_sentinel_gate(
    candidate_variant: str,
    *,
    champion_variant: str,
    scenario: str,
    item_rows: dict[tuple[str, str, str], dict[str, Any]],
    sentinel_file_ids: list[str],
    gate_cfg: dict[str, Any],
) -> dict[str, Any]:
    minimum_verdict = str(gate_cfg.get("minimum_verdict", "borderline"))
    deltas: list[float] = []
    rows: list[dict[str, Any]] = []
    verdict_failures: list[str] = []
    for file_id in sentinel_file_ids:
        candidate = item_rows.get((candidate_variant, scenario, file_id))
        champion = item_rows.get((champion_variant, scenario, file_id))
        if not candidate or not champion:
            return {"status": "pending_review", "reason": f"Missing sentinel review row for {file_id}."}
        candidate_score = _safe_float(candidate.get("weighted_score"))
        champion_score = _safe_float(champion.get("weighted_score"))
        delta = None if candidate_score is None or champion_score is None else round(candidate_score - champion_score, 2)
        if delta is not None:
            deltas.append(delta)
        candidate_verdict = str(candidate.get("verdict"))
        if not _verdict_meets(candidate_verdict, minimum_verdict):
            verdict_failures.append(file_id)
        rows.append(
            {
                "file_id": file_id,
                "candidate_score": candidate_score,
                "champion_score": champion_score,
                "candidate_verdict": candidate_verdict,
                "champion_verdict": champion.get("verdict"),
                "score_delta": delta,
            }
        )
    mean_delta = round(sum(deltas) / len(deltas), 2) if deltas else None
    improved = any(
        _verdict_rank(str(row["candidate_verdict"])) > _verdict_rank(str(row["champion_verdict"]))
        for row in rows
    )
    passed = not verdict_failures and (
        (mean_delta is not None and mean_delta >= float(gate_cfg.get("min_score_delta", 8.0))) or improved
    )
    return {
        "status": "pass" if passed else "fail",
        "rows": rows,
        "mean_score_delta": mean_delta,
        "minimum_verdict": minimum_verdict,
        "verdict_failures": verdict_failures,
    }


def _evaluate_throughput_gate(
    candidate_variant: str,
    *,
    champion_variant: str,
    cycle_spec: dict[str, Any],
    stage_run_root: Path | None,
    gate_cfg: dict[str, Any],
) -> dict[str, Any]:
    if stage_run_root is None:
        return {"status": "skipped", "reason": "No throughput stage configured."}
    throughput_cfg = dict(cycle_spec["datasets"].get("throughput", {}))
    scenario = _scenario_name(int(throughput_cfg.get("chunk_minutes", [30])[0]))
    max_penalty_pct = float(gate_cfg.get("max_throughput_penalty_pct", 25.0))
    requested_workers = [int(value) for value in throughput_cfg.get("throughput_workers", [1, 2, 4])]
    out_rows: list[dict[str, Any]] = []
    penalties: list[float] = []
    for worker in requested_workers:
        candidate_dir = _find_variant_artifact_dir(stage_run_root, candidate_variant)
        champion_dir = _find_variant_artifact_dir(stage_run_root, champion_variant)
        if candidate_dir is None or champion_dir is None:
            return {"status": "pending_review", "reason": "Missing throughput variant artifacts."}
        candidate_rows = _read_csv_rows(candidate_dir / "summary.csv")
        champion_rows = _read_csv_rows(champion_dir / "summary.csv")
        candidate_tp = _throughput_rows(candidate_rows, scenario).get(worker)
        champion_tp = _throughput_rows(champion_rows, scenario).get(worker)
        if not candidate_tp or not champion_tp:
            continue
        candidate_val = _safe_float(candidate_tp.get("throughput_audio_hours_per_hour"))
        champion_val = _safe_float(champion_tp.get("throughput_audio_hours_per_hour"))
        if candidate_val is None or champion_val is None or champion_val <= 0:
            continue
        penalty_pct = max(0.0, (champion_val - candidate_val) / champion_val * 100.0)
        penalties.append(penalty_pct)
        out_rows.append(
            {
                "worker": worker,
                "candidate_audio_hours_per_hour": round(candidate_val, 6),
                "champion_audio_hours_per_hour": round(champion_val, 6),
                "penalty_pct": round(penalty_pct, 2),
            }
        )
    if not out_rows:
        return {"status": "pending_review", "reason": "No usable throughput rows found."}
    max_penalty = round(max(penalties), 2)
    passed = max_penalty <= max_penalty_pct
    return {
        "status": "pass" if passed else "fail",
        "rows": out_rows,
        "max_penalty_pct": max_penalty,
        "max_allowed_penalty_pct": max_penalty_pct,
    }


def _evaluate_long_tail_gate(
    candidate_variant: str,
    *,
    champion_variant: str,
    scenario: str,
    grouped_rows: dict[tuple[str, str], dict[str, Any]],
    throughput_result: dict[str, Any],
    gate_cfg: dict[str, Any],
) -> dict[str, Any]:
    candidate = grouped_rows.get((candidate_variant, scenario))
    champion = grouped_rows.get((champion_variant, scenario))
    if not candidate or not champion:
        return {"status": "pending_review", "reason": "Missing long-tail review summary."}
    candidate_score = _safe_float(candidate.get("mean_weighted_score"))
    champion_score = _safe_float(champion.get("mean_weighted_score"))
    delta = None if candidate_score is None or champion_score is None else round(candidate_score - champion_score, 2)
    borderline_count = _safe_int(candidate.get("borderline_count")) or 0
    not_acceptable_count = _safe_int(candidate.get("not_acceptable_count")) or 0
    mean_improves = delta is not None and delta >= float(gate_cfg.get("min_mean_score_delta", 3.0))
    tie_window = float(gate_cfg.get("tie_break_window", 2.0))
    throughput_ok = throughput_result.get("status") in {"pass", "skipped"}
    same_band = str(candidate.get("overall_verdict")) == str(champion.get("overall_verdict"))
    quality_tie_break = (
        delta is not None
        and abs(delta) <= tie_window
        and same_band
        and (
            (_safe_int(candidate.get("not_acceptable_count")) or 0) < (_safe_int(champion.get("not_acceptable_count")) or 0)
            or (_safe_int(candidate.get("major_issues")) or 0) < (_safe_int(champion.get("major_issues")) or 0)
        )
    )
    passed = bool(
        throughput_ok
        and not_acceptable_count == 0
        and borderline_count <= int(gate_cfg.get("max_borderline_items", 1))
        and (mean_improves or quality_tie_break)
    )
    return {
        "status": "pass" if passed else "fail",
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "score_delta": delta,
        "candidate_verdict": candidate.get("overall_verdict"),
        "champion_verdict": champion.get("overall_verdict"),
        "borderline_count": borderline_count,
        "not_acceptable_count": not_acceptable_count,
        "quality_tie_break": quality_tie_break,
    }


def _stage_review_data(stage_run_root: Path | None) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    if stage_run_root is None:
        return {}, {}
    review_outputs = _ensure_review_summary(stage_run_root)
    if review_outputs is None:
        return {}, {}
    scoreboard_rows = _read_csv_rows(Path(review_outputs["scoreboard_csv"]))
    item_rows = _read_csv_rows(Path(review_outputs["item_scores_csv"]))
    return _group_stage_rows(scoreboard_rows), _item_stage_rows(item_rows)


def _update_backlog_statuses(program: dict[str, Any], decision_summary: dict[str, Any]) -> None:
    promoted = decision_summary.get("promoted_candidate")
    result_map = {item["id"]: item for item in decision_summary.get("candidate_results", [])}
    for item in program.get("backlog", []):
        item_id = item.get("id")
        if item_id == promoted:
            item["status"] = "promoted"
            continue
        result = result_map.get(item_id)
        if result is None:
            continue
        smoke_status = result.get("smoke", {}).get("status")
        sentinel_status = result.get("sentinel", {}).get("status")
        long_tail_status = result.get("long_tail", {}).get("status")
        if smoke_status == "fail" and result.get("smoke", {}).get("meaning_loss"):
            item["status"] = "killed"
        elif sentinel_status == "fail" and result.get("sentinel", {}).get("verdict_failures"):
            item["status"] = "deferred"
        elif long_tail_status == "pass":
            item["status"] = "ready"
        elif long_tail_status == "fail":
            item["status"] = "deferred"


def _evaluate_production_gate(program: dict[str, Any], champion_variant: str, long_tail_grouped: dict[tuple[str, str], dict[str, Any]], long_tail_items: dict[tuple[str, str, str], dict[str, Any]], cycle_spec: dict[str, Any], throughput_result: dict[str, Any]) -> bool:
    production_cfg = dict(program.get("production", {}))
    long_tail_cfg = dict(cycle_spec.get("datasets", {}).get("long_tail", {}))
    if not long_tail_cfg:
        return False
    scenario = _scenario_name(int(long_tail_cfg.get("chunk_minutes", [30])[0]))
    champion = long_tail_grouped.get((champion_variant, scenario))
    if not champion:
        return False
    if (_safe_float(champion.get("mean_weighted_score")) or 0.0) < float(production_cfg.get("min_mean_score", 85.0)):
        return False
    if (_safe_int(champion.get("critical_issues")) or 0) != 0:
        return False
    if (_safe_int(champion.get("not_acceptable_count")) or 0) != 0:
        return False
    meaning_loss = any(
        _parse_bool(row.get("meaning_loss"))
        for key, row in long_tail_items.items()
        if key[0] == champion_variant and key[1] == scenario
    )
    if meaning_loss:
        return False
    if throughput_result.get("status") not in {"pass", "skipped"}:
        return False
    return True


def _prepare(program_manifest: Path, cycle_id: str | None) -> dict[str, Any]:
    program = _read_json(program_manifest)
    program["_program_manifest"] = str(program_manifest)
    _validate_hook_policy(program)
    limit = int(program.get("selection", {}).get("max_candidates_per_cycle", 3))
    selected, ranked = select_cycle_candidates(list(program.get("backlog", [])), limit=limit)
    cycle_id = cycle_id or f"cycle_{_timestamp_slug().lower()}"
    cycle_spec = _build_cycle_spec(program, cycle_id, selected)
    cycle_root = Path(cycle_spec["cycle_root"])
    notes_paths = {name: Path(path) for name, path in cycle_spec["notes_paths"].items()}

    for item in program.get("backlog", []):
        if item.get("id") in {candidate["id"] for candidate in selected}:
            item["status"] = "active"
    for item in selected:
        item["status"] = "active"
    program["active_cycle_id"] = cycle_id
    program["updated_at"] = _utc_now_iso()

    _write_json(cycle_root / "cycle_spec.json", cycle_spec)
    _write_json(program_manifest, {k: v for k, v in program.items() if not k.startswith("_")})
    _write_text(notes_paths["program"], _render_program_note(program, ranked, selected))
    _write_text(notes_paths["backlog"], _render_backlog_note(ranked))
    _write_text(notes_paths["cycle"], _render_cycle_note(cycle_spec))
    return {
        "program_manifest": str(program_manifest),
        "cycle_spec": str(cycle_root / "cycle_spec.json"),
        "notes": {key: str(path) for key, path in notes_paths.items()},
    }


def _execute(cycle_spec_path: Path) -> dict[str, Any]:
    cycle_spec = _read_json(cycle_spec_path)
    non_runnable = [item["id"] for item in cycle_spec.get("selected_candidates", []) if not item.get("runnable_variant")]
    if non_runnable:
        raise RuntimeError(
            "Cannot execute cycle because selected ideas are not wired into the harness yet: "
            + ", ".join(non_runnable)
        )

    cycle_root = Path(cycle_spec["cycle_root"])
    cycle_root.mkdir(parents=True, exist_ok=True)
    stage_runs: dict[str, dict[str, Any]] = {}
    for stage_name in ("smoke", "sentinel", "long_tail", "throughput"):
        dataset_cfg = cycle_spec.get("datasets", {}).get(stage_name)
        if not dataset_cfg or not _stage_enabled(dataset_cfg):
            continue
        print(f"[INFO] Running iteration stage: {stage_name}")
        run_root = _run_stage(cycle_spec, stage_name)
        stage_runs[stage_name] = {
            "run_root": str(run_root),
            "review_json": str(run_root / "expert_review" / "review_template.json"),
            "scoreboard_csv": str(run_root / "expert_review" / "scoreboard.csv"),
            "item_scores_csv": str(run_root / "expert_review" / "item_scores.csv"),
        }

    state = {
        "cycle_id": cycle_spec["cycle_id"],
        "executed_at": _utc_now_iso(),
        "status": "executed",
        "stage_runs": stage_runs,
    }
    _write_json(cycle_root / "cycle_state.json", state)
    note_path = Path(cycle_spec["notes_paths"]["cycle"])
    _write_text(note_path, _render_cycle_note({**cycle_spec, "status": "executed"}))
    return state


def _finalize(cycle_spec_path: Path) -> dict[str, Any]:
    cycle_spec = _read_json(cycle_spec_path)
    program_manifest = Path(cycle_spec["program_manifest"]).resolve()
    program = _read_json(program_manifest)
    cycle_root = Path(cycle_spec["cycle_root"])
    cycle_state_path = cycle_root / "cycle_state.json"
    stage_runs = _read_json(cycle_state_path).get("stage_runs", {}) if cycle_state_path.exists() else {}

    champion_variant = str(cycle_spec["champion"]["variant"])
    sentinel_file_ids = list(program.get("current_findings", {}).get("sentinel_file_ids", []))
    smoke_grouped, smoke_items = _stage_review_data(Path(stage_runs["smoke"]["run_root"])) if "smoke" in stage_runs else ({}, {})
    sentinel_grouped, sentinel_items = _stage_review_data(Path(stage_runs["sentinel"]["run_root"])) if "sentinel" in stage_runs else ({}, {})
    long_tail_grouped, long_tail_items = _stage_review_data(Path(stage_runs["long_tail"]["run_root"])) if "long_tail" in stage_runs else ({}, {})
    throughput_stage_root = Path(stage_runs["throughput"]["run_root"]) if "throughput" in stage_runs else None

    smoke_scenario = _scenario_name(int(cycle_spec["datasets"]["smoke"].get("chunk_minutes", [10])[0])) if "smoke" in cycle_spec["datasets"] else ""
    sentinel_scenario = _scenario_name(int(cycle_spec["datasets"]["sentinel"].get("chunk_minutes", [30])[0])) if "sentinel" in cycle_spec["datasets"] else ""
    long_tail_scenario = _scenario_name(int(cycle_spec["datasets"]["long_tail"].get("chunk_minutes", [30])[0])) if "long_tail" in cycle_spec["datasets"] else ""
    throughput_gate_cfg = dict(cycle_spec.get("gates", {}).get("throughput", {}))

    candidate_results: list[dict[str, Any]] = []
    promoted_candidate: str | None = None
    best_candidate_score: float | None = None

    for candidate in cycle_spec.get("selected_candidates", []):
        if not candidate.get("runnable_variant"):
            candidate_results.append(
                {
                    "id": candidate["id"],
                    "smoke": {"status": "blocked"},
                    "sentinel": {"status": "blocked"},
                    "long_tail": {"status": "blocked"},
                    "throughput": {"status": "blocked"},
                    "overall_recommendation": "blocked_pending_implementation",
                }
            )
            continue
        variant = str(candidate["runnable_variant"])
        smoke_result = _evaluate_smoke_gate(
            candidate["id"],
            variant,
            champion_variant=champion_variant,
            scenario=smoke_scenario,
            grouped_rows=smoke_grouped,
            item_rows=smoke_items,
            gate_cfg=dict(cycle_spec.get("gates", {}).get("smoke", {})),
        )
        sentinel_result = _evaluate_sentinel_gate(
            variant,
            champion_variant=champion_variant,
            scenario=sentinel_scenario,
            item_rows=sentinel_items,
            sentinel_file_ids=sentinel_file_ids,
            gate_cfg=dict(cycle_spec.get("gates", {}).get("sentinel", {})),
        )
        throughput_result = _evaluate_throughput_gate(
            variant,
            champion_variant=champion_variant,
            cycle_spec=cycle_spec,
            stage_run_root=throughput_stage_root,
            gate_cfg=throughput_gate_cfg,
        )
        long_tail_result = _evaluate_long_tail_gate(
            variant,
            champion_variant=champion_variant,
            scenario=long_tail_scenario,
            grouped_rows=long_tail_grouped,
            throughput_result=throughput_result,
            gate_cfg=dict(cycle_spec.get("gates", {}).get("long_tail", {})),
        )

        if all(result.get("status") == "pass" for result in (smoke_result, sentinel_result, long_tail_result)):
            overall = "promote_candidate"
            candidate_score = _safe_float(long_tail_result.get("candidate_score"))
            if candidate_score is not None and (best_candidate_score is None or candidate_score > best_candidate_score):
                promoted_candidate = candidate["id"]
                best_candidate_score = candidate_score
        elif smoke_result.get("status") == "fail" and smoke_result.get("meaning_loss"):
            overall = "kill_candidate"
        elif sentinel_result.get("status") == "fail":
            overall = "defer_candidate"
        elif long_tail_result.get("status") == "fail":
            overall = "defer_candidate"
        else:
            overall = "pending_review"

        candidate_results.append(
            {
                "id": candidate["id"],
                "runnable_variant": variant,
                "smoke": smoke_result,
                "sentinel": sentinel_result,
                "long_tail": long_tail_result,
                "throughput": throughput_result,
                "overall_recommendation": overall,
            }
        )

    if promoted_candidate:
        promoted_item = next(item for item in cycle_spec["selected_candidates"] if item["id"] == promoted_candidate)
        program["champion"] = {
            "label": promoted_item["id"],
            "variant": promoted_item["runnable_variant"],
            "expert_score": next(
                (
                    result["long_tail"]["candidate_score"]
                    for result in candidate_results
                    if result["id"] == promoted_candidate
                ),
                None,
            ),
            "verdict": next(
                (
                    result["long_tail"]["candidate_verdict"]
                    for result in candidate_results
                    if result["id"] == promoted_candidate
                ),
                None,
            ),
            "source_cycle": cycle_spec["cycle_id"],
        }

    production_ready = _evaluate_production_gate(
        program,
        champion_variant=str(program.get("champion", {}).get("variant")),
        long_tail_grouped=long_tail_grouped,
        long_tail_items=long_tail_items,
        cycle_spec=cycle_spec,
        throughput_result=_evaluate_throughput_gate(
            str(program.get("champion", {}).get("variant")),
            champion_variant=str(program.get("champion", {}).get("variant")),
            cycle_spec=cycle_spec,
            stage_run_root=throughput_stage_root,
            gate_cfg=throughput_gate_cfg,
        ),
    )
    required_consecutive = int(program.get("production", {}).get("required_consecutive_passes", 2))
    if production_ready:
        program["consecutive_production_passes"] = int(program.get("consecutive_production_passes", 0)) + 1
    else:
        program["consecutive_production_passes"] = 0

    _update_backlog_statuses(program, {"promoted_candidate": promoted_candidate, "candidate_results": candidate_results})
    program["updated_at"] = _utc_now_iso()
    program["active_cycle_id"] = None

    decision_summary = {
        "cycle_id": cycle_spec["cycle_id"],
        "generated_at": _utc_now_iso(),
        "status": "finalized",
        "promoted_candidate": promoted_candidate,
        "candidate_results": candidate_results,
        "production_ready": bool(
            production_ready and int(program.get("consecutive_production_passes", 0)) >= required_consecutive
        ),
        "consecutive_production_passes": int(program.get("consecutive_production_passes", 0)),
        "required_consecutive_production_passes": required_consecutive,
    }

    _write_json(program_manifest, program)
    _write_json(cycle_root / "decision_summary.json", decision_summary)

    selected_candidates, ranked_backlog = select_cycle_candidates(
        list(program.get("backlog", [])),
        int(program.get("selection", {}).get("max_candidates_per_cycle", 3)),
    )
    notes_dir = Path(cycle_spec["notes_dir"])
    _write_text(notes_dir / "asr_iteration_program.md", _render_program_note(program, ranked_backlog, selected_candidates))
    _write_text(notes_dir / "asr_hypothesis_backlog.md", _render_backlog_note(ranked_backlog))
    _write_text(Path(cycle_spec["notes_paths"]["cycle"]), _render_cycle_note(cycle_spec, decision_summary))
    return decision_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage continuous ASR iteration cycles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Rank backlog, emit cycle spec, and refresh notes.")
    prepare.add_argument("--program-manifest", default=DEFAULT_PROGRAM_MANIFEST)
    prepare.add_argument("--cycle-id", default=None)

    execute = subparsers.add_parser("execute", help="Execute the validation ladder for a prepared cycle.")
    execute.add_argument("--cycle-spec", required=True)

    finalize = subparsers.add_parser("finalize", help="Finalize a cycle after expert review has been scored.")
    finalize.add_argument("--cycle-spec", required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        outputs = _prepare(Path(args.program_manifest).resolve(), args.cycle_id)
        print("[INFO] Iteration cycle prepared.")
        for key, value in outputs.items():
            print(f"[INFO] {key}: {value}")
        return
    if args.command == "execute":
        outputs = _execute(Path(args.cycle_spec).resolve())
        print("[INFO] Iteration cycle executed.")
        print(json.dumps(outputs, ensure_ascii=True, indent=2))
        return
    if args.command == "finalize":
        outputs = _finalize(Path(args.cycle_spec).resolve())
        print("[INFO] Iteration cycle finalized.")
        print(json.dumps(outputs, ensure_ascii=True, indent=2))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
