from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "experiments" / "expert_review.py"
)
SPEC = importlib.util.spec_from_file_location("expert_review", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

compute_weighted_score = MODULE.compute_weighted_score
derive_verdict = MODULE.derive_verdict
summarize_review_payload = MODULE.summarize_review_payload


def test_compute_weighted_score_perfect() -> None:
    scores = {
        "semantic_fidelity": 5,
        "critical_details": 5,
        "completeness": 5,
        "boundary_coherence": 5,
        "readability": 5,
        "noise_handling": 5,
    }
    assert compute_weighted_score(scores) == 100.0


def test_critical_issue_forces_not_acceptable() -> None:
    verdict = derive_verdict(92.0, {"critical": 1, "major": 0, "minor": 0}, False)
    assert verdict == "not_acceptable"


def test_summary_groups_and_ranks_rows() -> None:
    payload = {
        "rubric_version": "2026-04-09",
        "group_by_fields": ["variant"],
        "items": [
            {
                "comparison": {"variant": "baseline", "file_id": "call_1", "scenario": "chunk_30m"},
                "metrics": {"word_drift": 0.07, "rtf": 0.16},
                "review": {
                    "scores": {
                        "semantic_fidelity": 5,
                        "critical_details": 4,
                        "completeness": 4,
                        "boundary_coherence": 4,
                        "readability": 4,
                        "noise_handling": 4,
                    },
                    "severity_counts": {"critical": 0, "major": 0, "minor": 2},
                    "meaning_loss": False,
                    "verdict": None,
                    "summary": "Strong result.",
                    "reviewer": "test",
                    "reviewed_at": "2026-04-09T00:00:00+00:00",
                },
            },
            {
                "comparison": {"variant": "idea4", "file_id": "call_1", "scenario": "chunk_30m"},
                "metrics": {"word_drift": 0.09, "rtf": 0.12},
                "review": {
                    "scores": {
                        "semantic_fidelity": 2,
                        "critical_details": 2,
                        "completeness": 3,
                        "boundary_coherence": 2,
                        "readability": 2,
                        "noise_handling": 3,
                    },
                    "severity_counts": {"critical": 1, "major": 1, "minor": 0},
                    "meaning_loss": True,
                    "verdict": None,
                    "summary": "Meaning-changing errors.",
                    "reviewer": "test",
                    "reviewed_at": "2026-04-09T00:00:00+00:00",
                },
            },
        ],
    }

    summary = summarize_review_payload(payload)
    assert len(summary["group_rows"]) == 2
    assert summary["group_rows"][0]["variant"] == "baseline"
    assert summary["group_rows"][0]["overall_verdict"] in {"acceptable", "equivalent"}
    assert summary["group_rows"][1]["variant"] == "idea4"
    assert summary["group_rows"][1]["overall_verdict"] == "not_acceptable"


def test_summary_excludes_bad_gold_from_group_ranking() -> None:
    payload = {
        "rubric_version": "2026-04-09",
        "group_by_fields": ["variant"],
        "items": [
            {
                "comparison": {"variant": "baseline", "file_id": "call_valid", "scenario": "chunk_30m"},
                "metrics": {"word_drift": 0.07, "rtf": 0.16},
                "review": {
                    "exclude_from_ranking": False,
                    "exclusion_reason": None,
                    "scores": {
                        "semantic_fidelity": 5,
                        "critical_details": 4,
                        "completeness": 4,
                        "boundary_coherence": 4,
                        "readability": 4,
                        "noise_handling": 4,
                    },
                    "severity_counts": {"critical": 0, "major": 0, "minor": 1},
                    "meaning_loss": False,
                    "verdict": None,
                    "summary": "Valid scored item.",
                    "reviewer": "test",
                    "reviewed_at": "2026-04-09T00:00:00+00:00",
                },
            },
            {
                "comparison": {"variant": "baseline", "file_id": "call_bad_gold", "scenario": "chunk_30m"},
                "metrics": {"word_drift": 0.99, "rtf": 0.30},
                "review": {
                    "exclude_from_ranking": True,
                    "exclusion_reason": "Gold transcript is truncated and repeats filler.",
                    "scores": {
                        "semantic_fidelity": None,
                        "critical_details": None,
                        "completeness": None,
                        "boundary_coherence": None,
                        "readability": None,
                        "noise_handling": None,
                    },
                    "severity_counts": {"critical": 0, "major": 0, "minor": 0},
                    "meaning_loss": None,
                    "verdict": None,
                    "summary": "Excluded.",
                    "reviewer": "test",
                    "reviewed_at": "2026-04-09T00:00:00+00:00",
                },
            },
        ],
    }

    summary = summarize_review_payload(payload)
    assert len(summary["group_rows"]) == 1
    group = summary["group_rows"][0]
    assert group["variant"] == "baseline"
    assert group["reviewed_items"] == 1
    assert group["excluded_items"] == 1
    assert group["total_items"] == 2
    assert group["mean_metric_word_drift"] == 0.07
    assert group["overall_verdict"] in {"acceptable", "equivalent"}
    excluded_row = next(row for row in summary["item_rows"] if row["file_id"] == "call_bad_gold")
    assert excluded_row["verdict"] == "excluded"
    assert excluded_row["weighted_score"] is None
    assert excluded_row["exclude_from_ranking"] is True
