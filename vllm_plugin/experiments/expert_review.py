#!/usr/bin/env python3
"""
Shared rubric and artifact helpers for expert transcript review.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


RUBRIC_VERSION = "2026-04-09"
SCORE_MIN = 0
SCORE_MAX = 5
ISSUE_CATEGORIES = [
    "meaning_change",
    "entity_name",
    "number_or_amount",
    "date_or_time",
    "consent_or_verification",
    "omission",
    "addition",
    "boundary_glitch",
    "noise_or_tagging",
    "readability",
    "other",
]


@dataclass(frozen=True)
class Criterion:
    key: str
    title: str
    weight: int
    prompt: str
    anchors: dict[int, str]


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="semantic_fidelity",
        title="Semantic Fidelity",
        weight=35,
        prompt="Does the candidate preserve the same intent, decisions, and real-world meaning?",
        anchors={
            5: "Meaning is preserved end-to-end. Only cosmetic wording differences.",
            4: "Minor wording drift, but no business-relevant meaning change.",
            3: "Noticeable paraphrase or mild ambiguity, but main meaning still holds.",
            2: "Meaning is degraded in important stretches or intent becomes less reliable.",
            1: "Multiple meaning-changing substitutions or omissions.",
            0: "Transcript is unusable for semantic fidelity.",
        },
    ),
    Criterion(
        key="critical_details",
        title="Critical Details",
        weight=25,
        prompt=(
            "Are names, providers, numbers, amounts, dates, consent language, and "
            "verification/disposition details preserved accurately?"
        ),
        anchors={
            5: "Critical entities and details are consistently reliable.",
            4: "One or two minor detail defects, but no likely business impact.",
            3: "Some detail drift; manual spot-check may be required.",
            2: "Important detail reliability is weak or inconsistent.",
            1: "Critical details are frequently wrong or unstable.",
            0: "Critical details cannot be trusted.",
        },
    ),
    Criterion(
        key="completeness",
        title="Completeness",
        weight=15,
        prompt="Does the candidate avoid material omissions, premature truncation, or invented content?",
        anchors={
            5: "No meaningful omissions or additions.",
            4: "Minor drops/additions only; meaning still complete.",
            3: "Noticeable omissions/additions, but main arc remains intact.",
            2: "Material portions are lost or spuriously added.",
            1: "Large omissions/additions distort the call record.",
            0: "Completeness is unacceptable.",
        },
    ),
    Criterion(
        key="boundary_coherence",
        title="Boundary Coherence",
        weight=10,
        prompt="Are chunk seams smooth, without duplicated, clipped, or contradictory text?",
        anchors={
            5: "Seams are effectively invisible.",
            4: "Minor seam artifacts, not distracting.",
            3: "Seams are visible, but mostly tolerable.",
            2: "Repeated or clipped seam text affects trust.",
            1: "Boundary behavior regularly damages meaning.",
            0: "Boundary stitching fails badly.",
        },
    ),
    Criterion(
        key="readability",
        title="Readability",
        weight=10,
        prompt="Is the transcript clean and easy for an operator or QA reviewer to read?",
        anchors={
            5: "Reads naturally and cleanly.",
            4: "Readable with only small awkwardness.",
            3: "Readable, but noticeably noisy or fragmented.",
            2: "Frequent awkwardness slows understanding.",
            1: "Hard to read reliably.",
            0: "Operationally unreadable.",
        },
    ),
    Criterion(
        key="noise_handling",
        title="Noise Handling",
        weight=5,
        prompt="Are silence/noise/unintelligible markers used appropriately and not overproduced?",
        anchors={
            5: "Noise tags are sparse and appropriate.",
            4: "Minor tagging issues only.",
            3: "Tagging is somewhat noisy, but acceptable.",
            2: "Tagging or filler is distracting.",
            1: "Noise handling frequently degrades readability.",
            0: "Noise handling is unusable.",
        },
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def rubric_definition() -> dict[str, Any]:
    return {
        "rubric_version": RUBRIC_VERSION,
        "score_range": {"min": SCORE_MIN, "max": SCORE_MAX},
        "criteria": [
            {
                "key": criterion.key,
                "title": criterion.title,
                "weight": criterion.weight,
                "prompt": criterion.prompt,
                "anchors": {str(score): text for score, text in criterion.anchors.items()},
            }
            for criterion in CRITERIA
        ],
        "severity_scale": {
            "critical": "Meaning-changing or business-impacting error. Any critical issue makes the transcript not acceptable.",
            "major": "Important defect that weakens trust, but does not fully overturn the call meaning.",
            "minor": "Cosmetic or low-impact defect.",
        },
        "issue_categories": ISSUE_CATEGORIES,
        "verdict_bands": {
            "equivalent": "Score >= 90, no critical issues, and no material meaning loss.",
            "acceptable": "Score >= 80, no critical issues, and meaning preserved.",
            "borderline": "Score >= 70 with no critical issues, but noticeable quality concerns remain.",
            "not_acceptable": "Score < 70, or any critical issue, or material meaning loss.",
        },
        "review_instructions": [
            "Review the candidate against the gold transcript first; listen to audio only if textual evidence is ambiguous.",
            "Score each criterion from 0 to 5 using whole numbers.",
            "Count issues by severity and capture at least one example for each major or critical defect.",
            "Prefer business meaning over cosmetic wording. Punctuation differences alone should not move the score.",
            "If the gold transcript is clearly invalid or unusable, set `exclude_from_ranking` to true and record `exclusion_reason` instead of forcing a score.",
        ],
    }


def render_rubric_markdown() -> str:
    lines = [
        "# Expert Transcript Evaluation Rubric",
        "",
        f"Rubric version: `{RUBRIC_VERSION}`",
        "",
        "Use this rubric when deciding whether a chunking method is good enough to replace the baseline in production-facing ASR workflows.",
        "",
        "## Scoring",
        "",
        f"- Score each criterion from `{SCORE_MIN}` to `{SCORE_MAX}` using whole numbers.",
        "- The final weighted score is normalized to a 0-100 scale.",
        "- Any critical issue or explicit meaning loss forces the verdict to `not_acceptable`.",
        "- If the gold transcript is invalid, exclude the item from ranking instead of fabricating a score.",
        "",
        "## Criteria",
        "",
    ]
    for criterion in CRITERIA:
        lines.append(f"### {criterion.title} ({criterion.weight}%)")
        lines.append("")
        lines.append(f"- Prompt: {criterion.prompt}")
        for score in range(SCORE_MAX, SCORE_MIN - 1, -1):
            lines.append(f"- `{score}`: {criterion.anchors[score]}")
        lines.append("")

    lines.extend(
        [
            "## Severity Levels",
            "",
            "- `critical`: meaning-changing or business-impacting defect. Examples: wrong provider, wrong amount, wrong consent/disposition, inversion of meaning.",
            "- `major`: important quality problem that weakens trust but does not clearly invert meaning.",
            "- `minor`: cosmetic or low-impact issue.",
            "",
            "## Recommended Workflow",
            "",
            "1. Read the gold and candidate transcripts side by side.",
            "2. Inspect seam regions and any metric-flagged drift segments.",
            "3. Score each criterion independently.",
            "4. Record severity counts and concrete issue examples.",
            "5. If the gold transcript is invalid, set `exclude_from_ranking` to true and record `exclusion_reason`.",
            "6. Otherwise assign the verdict based on the rubric bands, with critical issues overriding the numeric score.",
            "",
            "## Issue Categories",
            "",
            "- " + "\n- ".join(f"`{name}`" for name in ISSUE_CATEGORIES),
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def make_review_item(
    comparison: Mapping[str, Any],
    gold_txt_path: Path,
    candidate_txt_path: Path,
    metrics: Mapping[str, Any] | None = None,
    diff_snippets: Sequence[str] | None = None,
) -> dict[str, Any]:
    review_id_parts = []
    for key in ("endpoint", "variant", "file_id", "scenario"):
        value = comparison.get(key)
        if value not in (None, ""):
            review_id_parts.append(f"{key}={value}")
    review_id = " | ".join(review_id_parts) or "review_item"
    return {
        "review_id": review_id,
        "comparison": dict(comparison),
        "gold_transcript": {
            "txt_path": str(gold_txt_path),
            "json_path": str(gold_txt_path.with_suffix(".json")),
        },
        "candidate_transcript": {
            "txt_path": str(candidate_txt_path),
            "json_path": str(candidate_txt_path.with_suffix(".json")),
        },
        "metrics": dict(metrics or {}),
        "diff_snippets": list(diff_snippets or []),
        "review": {
            "reviewer": None,
            "reviewed_at": None,
            "exclude_from_ranking": False,
            "exclusion_reason": None,
            "scores": {criterion.key: None for criterion in CRITERIA},
            "severity_counts": {"critical": 0, "major": 0, "minor": 0},
            "meaning_loss": None,
            "verdict": None,
            "summary": None,
            "notes": [],
            "issue_examples": [],
        },
    }


def write_review_bundle(
    output_dir: Path,
    *,
    title: str,
    items: Sequence[dict[str, Any]],
    group_by_fields: Sequence[str],
    source_kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    review_dir = output_dir / "expert_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    rubric_md_path = review_dir / "rubric.md"
    rubric_json_path = review_dir / "rubric.json"
    template_json_path = review_dir / "review_template.json"
    template_md_path = review_dir / "review_template.md"

    payload = {
        "title": title,
        "rubric_version": RUBRIC_VERSION,
        "generated_at": _utc_now_iso(),
        "source_kind": source_kind,
        "group_by_fields": list(group_by_fields),
        "metadata": dict(metadata or {}),
        "items": list(items),
    }

    _write_text(rubric_md_path, render_rubric_markdown())
    _write_json(rubric_json_path, rubric_definition())
    _write_json(template_json_path, payload)
    _write_text(template_md_path, render_review_template_markdown(payload))
    return {
        "review_dir": review_dir,
        "rubric_md": rubric_md_path,
        "rubric_json": rubric_json_path,
        "template_json": template_json_path,
        "template_md": template_md_path,
    }


def render_review_template_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload.get('title', 'Expert Transcript Review Template')}",
        "",
        f"Rubric version: `{payload.get('rubric_version', RUBRIC_VERSION)}`",
        "",
        "Fill `review_template.json`, then run:",
        "",
        "```bash",
        "python3 vllm_plugin/experiments/summarize_expert_review.py --review-json <path/to/review_template.json>",
        "```",
        "",
        "## Items",
        "",
    ]
    for item in payload.get("items", []):
        comparison = item.get("comparison", {})
        lines.append(f"### {item.get('review_id', 'review_item')}")
        lines.append("")
        for key in ("endpoint", "variant", "file_id", "scenario"):
            value = comparison.get(key)
            if value not in (None, ""):
                lines.append(f"- {key}: `{value}`")
        lines.append(f"- Gold txt: `{item['gold_transcript']['txt_path']}`")
        lines.append(f"- Candidate txt: `{item['candidate_transcript']['txt_path']}`")
        metrics = item.get("metrics", {})
        if metrics:
            metric_bits = []
            for key in (
                "word_drift",
                "seam_word_drift",
                "char_drift",
                "boundary_mae_sec",
                "rtf",
                "wall_time_sec",
            ):
                if metrics.get(key) is not None:
                    metric_bits.append(f"{key}={metrics[key]}")
            if metric_bits:
                lines.append(f"- Metrics: {', '.join(metric_bits)}")
        diff_snippets = item.get("diff_snippets", [])
        if diff_snippets:
            lines.append("- Diff hints:")
            for snippet in diff_snippets:
                lines.append(f"  - {snippet}")
        lines.append("- Review fields:")
        lines.append("  - `exclude_from_ranking`: ")
        lines.append("  - `exclusion_reason`: ")
        for criterion in CRITERIA:
            lines.append(f"  - `{criterion.key}`: ")
        lines.append("  - `critical/major/minor`: ")
        lines.append("  - `meaning_loss`: ")
        lines.append("  - `verdict`: ")
        lines.append("  - `summary`: ")
        lines.append("")
    return "\n".join(lines) + "\n"


def compute_weighted_score(scores: Mapping[str, Any]) -> float | None:
    total = 0.0
    weight_total = 0
    for criterion in CRITERIA:
        value = scores.get(criterion.key)
        if value is None:
            return None
        score = float(value)
        if score < SCORE_MIN or score > SCORE_MAX:
            raise ValueError(
                f"Invalid score for {criterion.key}: {score}. Expected {SCORE_MIN}-{SCORE_MAX}."
            )
        total += score * criterion.weight
        weight_total += criterion.weight
    if weight_total == 0:
        return None
    return round((total / (SCORE_MAX * weight_total)) * 100.0, 2)


def derive_verdict(
    weighted_score: float | None,
    severity_counts: Mapping[str, Any] | None,
    meaning_loss: bool | None,
) -> str | None:
    if weighted_score is None:
        return None
    critical = int((severity_counts or {}).get("critical", 0) or 0)
    if critical > 0 or meaning_loss:
        return "not_acceptable"
    if weighted_score >= 90.0:
        return "equivalent"
    if weighted_score >= 80.0:
        return "acceptable"
    if weighted_score >= 70.0:
        return "borderline"
    return "not_acceptable"


def summarize_review_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    group_by = [str(field) for field in payload.get("group_by_fields", [])]
    item_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    for item in payload.get("items", []):
        comparison = dict(item.get("comparison", {}))
        review = dict(item.get("review", {}))
        excluded = bool(review.get("exclude_from_ranking"))
        exclusion_reason = review.get("exclusion_reason")
        scores = dict(review.get("scores", {}))
        severity_counts = dict(review.get("severity_counts", {}))
        meaning_loss = review.get("meaning_loss")
        weighted_score = None if excluded else (compute_weighted_score(scores) if scores else None)
        verdict = (
            "excluded"
            if excluded
            else (review.get("verdict") or derive_verdict(weighted_score, severity_counts, meaning_loss))
        )
        row = {
            **comparison,
            **{
                f"metric_{key}": value
                for key, value in dict(item.get("metrics", {})).items()
            },
            **{f"score_{criterion.key}": scores.get(criterion.key) for criterion in CRITERIA},
            "critical_issues": int(severity_counts.get("critical", 0) or 0),
            "major_issues": int(severity_counts.get("major", 0) or 0),
            "minor_issues": int(severity_counts.get("minor", 0) or 0),
            "exclude_from_ranking": excluded,
            "exclusion_reason": exclusion_reason,
            "meaning_loss": meaning_loss,
            "weighted_score": weighted_score,
            "verdict": verdict,
            "summary": review.get("summary"),
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
        }
        item_rows.append(row)

        key = tuple(str(comparison.get(field, "")) for field in group_by)
        grouped.setdefault(key, []).append(row)

    group_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        included_rows = [row for row in rows if not bool(row.get("exclude_from_ranking"))]
        scores = [
            float(row["weighted_score"])
            for row in included_rows
            if row.get("weighted_score") is not None
        ]
        verdict_counts: dict[str, int] = {}
        for row in included_rows:
            verdict = str(row.get("verdict") or "unreviewed")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        group_row = {
            field: value for field, value in zip(group_by, key)
        }
        group_row.update(
            {
                "total_items": len(rows),
                "reviewed_items": len(scores),
                "excluded_items": sum(1 for row in rows if bool(row.get("exclude_from_ranking"))),
                "mean_weighted_score": round(sum(scores) / len(scores), 2) if scores else None,
                "critical_issues": sum(int(row.get("critical_issues", 0) or 0) for row in included_rows),
                "major_issues": sum(int(row.get("major_issues", 0) or 0) for row in included_rows),
                "minor_issues": sum(int(row.get("minor_issues", 0) or 0) for row in included_rows),
                "equivalent_count": verdict_counts.get("equivalent", 0),
                "acceptable_count": verdict_counts.get("acceptable", 0),
                "borderline_count": verdict_counts.get("borderline", 0),
                "not_acceptable_count": verdict_counts.get("not_acceptable", 0),
                "mean_metric_word_drift": _mean_metric(included_rows, "metric_word_drift"),
                "mean_metric_char_drift": _mean_metric(included_rows, "metric_char_drift"),
                "mean_metric_rtf": _mean_metric(included_rows, "metric_rtf"),
            }
        )
        group_row["overall_verdict"] = (
            derive_verdict(
                weighted_score=group_row["mean_weighted_score"],
                severity_counts={"critical": group_row["critical_issues"]},
                meaning_loss=group_row["not_acceptable_count"] > 0,
            )
            if scores
            else "insufficient_review"
        )
        group_rows.append(group_row)

    group_rows.sort(
        key=lambda row: (
            -float(row.get("mean_weighted_score") or 0.0),
            int(row.get("critical_issues") or 0),
            int(row.get("major_issues") or 0),
            float(row.get("mean_metric_word_drift") or 999.0),
        )
    )

    return {
        "rubric_version": payload.get("rubric_version", RUBRIC_VERSION),
        "generated_at": _utc_now_iso(),
        "group_by_fields": group_by,
        "item_rows": item_rows,
        "group_rows": group_rows,
    }


def write_review_summary(
    output_dir: Path,
    summary_payload: Mapping[str, Any],
) -> dict[str, Path]:
    item_csv_path = output_dir / "item_scores.csv"
    group_csv_path = output_dir / "scoreboard.csv"
    group_md_path = output_dir / "scoreboard.md"
    summary_json_path = output_dir / "summary.json"

    _write_csv(item_csv_path, list(summary_payload.get("item_rows", [])))
    _write_csv(group_csv_path, list(summary_payload.get("group_rows", [])))
    _write_text(group_md_path, render_scoreboard_markdown(summary_payload))
    _write_json(summary_json_path, summary_payload)
    return {
        "item_csv": item_csv_path,
        "scoreboard_csv": group_csv_path,
        "scoreboard_md": group_md_path,
        "summary_json": summary_json_path,
    }


def render_scoreboard_markdown(summary_payload: Mapping[str, Any]) -> str:
    group_by = list(summary_payload.get("group_by_fields", []))
    title = " / ".join(group_by) if group_by else "review group"
    lines = [
        "# Expert Review Scoreboard",
        "",
        f"Grouped by: `{title}`",
        "",
        "| Group | Reviewed | Excluded | Mean score | Verdict | Critical | Major | Minor | Equivalent | Acceptable | Borderline | Not acceptable | Mean drift | Mean RTF |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_payload.get("group_rows", []):
        label = ", ".join(f"{field}={row.get(field)}" for field in group_by) if group_by else "all"
        drift = row.get("mean_metric_word_drift")
        rtf = row.get("mean_metric_rtf")
        lines.append(
            "| {label} | {reviewed} | {excluded} | {score} | {verdict} | {critical} | {major} | {minor} | {equiv} | {acc} | {border} | {bad} | {drift} | {rtf} |".format(
                label=label,
                reviewed=int(row.get("reviewed_items") or 0),
                excluded=int(row.get("excluded_items") or 0),
                score=(
                    f"{float(row.get('mean_weighted_score')):.2f}"
                    if row.get("mean_weighted_score") is not None
                    else "NA"
                ),
                verdict=row.get("overall_verdict"),
                critical=int(row.get("critical_issues") or 0),
                major=int(row.get("major_issues") or 0),
                minor=int(row.get("minor_issues") or 0),
                equiv=int(row.get("equivalent_count") or 0),
                acc=int(row.get("acceptable_count") or 0),
                border=int(row.get("borderline_count") or 0),
                bad=int(row.get("not_acceptable_count") or 0),
                drift=f"{float(drift):.4f}" if drift is not None else "NA",
                rtf=f"{float(rtf):.4f}" if rtf is not None else "NA",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
