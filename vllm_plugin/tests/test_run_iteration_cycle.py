from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "run_iteration_cycle.py"
SPEC = importlib.util.spec_from_file_location("run_iteration_cycle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

compute_priority_score = MODULE.compute_priority_score
select_cycle_candidates = MODULE.select_cycle_candidates
evaluate_sentinel_gate = MODULE._evaluate_sentinel_gate
validate_hook_policy = MODULE._validate_hook_policy


def test_compute_priority_score_matches_program_formula() -> None:
    item = {
        "impact": 5,
        "failure_severity": 5,
        "confidence": 4,
        "reusability": 5,
        "effort": 3,
        "runtime_cost": 2,
    }
    assert compute_priority_score(item) == 29


def test_select_cycle_candidates_prefers_strategy_order() -> None:
    backlog = [
        {
            "id": "idea6_coverage_first_repair",
            "status": "planned",
            "impact": 5,
            "failure_severity": 5,
            "confidence": 4,
            "reusability": 5,
            "effort": 3,
            "runtime_cost": 2,
            "target_failure_modes": ["completeness_loss", "late_call_dropout"],
        },
        {
            "id": "idea8_language_aware_translator_mode",
            "status": "planned",
            "impact": 5,
            "failure_severity": 4,
            "confidence": 4,
            "reusability": 4,
            "effort": 3,
            "runtime_cost": 1,
            "target_failure_modes": ["mixed_language_stability", "entity_and_number_drift"],
        },
        {
            "id": "idea7_tail_rescue_backward_pass",
            "status": "planned",
            "impact": 4,
            "failure_severity": 4,
            "confidence": 4,
            "reusability": 4,
            "effort": 2,
            "runtime_cost": 2,
            "target_failure_modes": ["late_call_dropout", "completeness_loss"],
        },
        {
            "id": "idea9_degeneracy_guard",
            "status": "planned",
            "impact": 4,
            "failure_severity": 4,
            "confidence": 3,
            "reusability": 4,
            "effort": 2,
            "runtime_cost": 1,
            "target_failure_modes": ["degeneracy_or_looping", "completeness_loss"],
        },
    ]

    selected, ranked = select_cycle_candidates(backlog, limit=3)

    assert [item["id"] for item in selected] == [
        "idea6_coverage_first_repair",
        "idea8_language_aware_translator_mode",
        "idea7_tail_rescue_backward_pass",
    ]
    assert ranked[0]["priority_score"] == 29


def test_sentinel_gate_requires_borderline_or_better() -> None:
    item_rows = {
        ("candidate", "chunk_30m_overlap_context", "sentinel_a"): {
            "weighted_score": "82",
            "verdict": "acceptable",
        },
        ("candidate", "chunk_30m_overlap_context", "sentinel_b"): {
            "weighted_score": "74",
            "verdict": "borderline",
        },
        ("champion", "chunk_30m_overlap_context", "sentinel_a"): {
            "weighted_score": "70",
            "verdict": "borderline",
        },
        ("champion", "chunk_30m_overlap_context", "sentinel_b"): {
            "weighted_score": "64",
            "verdict": "not_acceptable",
        },
    }

    result = evaluate_sentinel_gate(
        "candidate",
        champion_variant="champion",
        scenario="chunk_30m_overlap_context",
        item_rows=item_rows,
        sentinel_file_ids=["sentinel_a", "sentinel_b"],
        gate_cfg={"minimum_verdict": "borderline", "min_score_delta": 8.0},
    )

    assert result["status"] == "pass"
    assert result["mean_score_delta"] == 11.0
    assert result["verdict_failures"] == []


def test_hook_policy_validation_rejects_missing_runnable_variant() -> None:
    program = {
        "hook_policy": {
            "required_backlog_fields": ["id", "title", "runnable_variant"],
        },
        "backlog": [
            {"id": "idea6", "title": "Coverage", "status": "active", "runnable_variant": "idea6"},
            {"id": "idea7", "title": "Tail rescue", "status": "active", "runnable_variant": None},
            {"id": "idea9", "title": "Planned", "status": "planned", "runnable_variant": None},
        ],
    }
    try:
        validate_hook_policy(program)
        assert False, "Expected ValueError for missing runnable_variant"
    except ValueError as exc:
        assert "idea7" in str(exc)
        assert "runnable_variant" in str(exc)
