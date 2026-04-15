#!/usr/bin/env python3
"""
Autonomous supervisor for an iteration cycle.

Behavior:
- Polls cycle state every N seconds (default 900 = 15 minutes)
- Ensures execute is running until the cycle emits cycle_state.json
- Runs expert-review summarization for completed stages when score data exists
- Finalizes the cycle after execute completes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_execute_pids(cycle_spec_path: Path) -> list[int]:
    cmd = [
        "pgrep",
        "-f",
        f"run_iteration_cycle.py execute --cycle-spec {cycle_spec_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _has_review_scores(review_json_path: Path) -> bool:
    if not review_json_path.exists():
        return False
    payload = _read_json(review_json_path)
    for item in payload.get("items", []):
        review = dict(item.get("review", {}))
        if bool(review.get("exclude_from_ranking")):
            return True
        scores = dict(review.get("scores", {}))
        if any(value not in (None, "") for value in scores.values()):
            return True
    return False


def _run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"\n[{_utc_now_iso()}] RUN: {' '.join(cmd)}\n")
        fp.flush()
        proc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, text=True, check=False)
        fp.write(f"[{_utc_now_iso()}] EXIT {proc.returncode}\n")
        fp.flush()
        return proc.returncode


def _start_execute(cycle_spec_path: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"\n[{_utc_now_iso()}] Starting execute subprocess.\n")
        fp.flush()
        proc = subprocess.Popen(
            [
                "python3",
                "-u",
                "vllm_plugin/experiments/run_iteration_cycle.py",
                "execute",
                "--cycle-spec",
                str(cycle_spec_path),
            ],
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        fp.write(f"[{_utc_now_iso()}] Started execute pid={proc.pid}\n")
        fp.flush()
        return int(proc.pid)


def _list_stage_roots(cycle_root: Path) -> dict[str, Path]:
    stages_root = cycle_root / "stages"
    out: dict[str, Path] = {}
    if not stages_root.exists():
        return out
    for stage_dir in sorted(stages_root.iterdir()):
        if not stage_dir.is_dir():
            continue
        runs = sorted([path for path in stage_dir.iterdir() if path.is_dir()])
        if runs:
            out[stage_dir.name] = runs[-1]
    return out


def _summarize_stage_if_ready(stage_root: Path, log_path: Path) -> bool:
    review_json = stage_root / "expert_review" / "review_template.json"
    scoreboard_csv = stage_root / "expert_review" / "scoreboard.csv"
    if not review_json.exists():
        return False
    if scoreboard_csv.exists():
        return True
    if not _has_review_scores(review_json):
        return False
    rc = _run_command(
        [
            "python3",
            "vllm_plugin/experiments/summarize_expert_review.py",
            "--review-json",
            str(review_json),
        ],
        log_path,
    )
    return rc == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervise an iteration cycle autonomously.")
    parser.add_argument("--cycle-spec", required=True, help="Path to cycle_spec.json")
    parser.add_argument("--poll-seconds", type=int, default=900, help="Polling interval in seconds.")
    parser.add_argument("--max-hours", type=float, default=8.0, help="Maximum supervision window.")
    parser.add_argument(
        "--log-path",
        default=None,
        help="Supervisor log file path. Defaults to <cycle_root>/logs/supervisor.log",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cycle_spec_path = Path(args.cycle_spec).resolve()
    cycle_spec = _read_json(cycle_spec_path)
    cycle_root = Path(cycle_spec["cycle_root"]).resolve()
    log_path = (
        Path(args.log_path).resolve()
        if args.log_path
        else (cycle_root / "logs" / "supervisor.log").resolve()
    )
    state_path = cycle_root / "supervisor_state.json"
    execute_log = cycle_root / "logs" / "execute_cycle.log"

    deadline = datetime.now(timezone.utc) + timedelta(hours=float(args.max_hours))
    state: dict[str, Any] = {
        "cycle_spec": str(cycle_spec_path),
        "started_at": _utc_now_iso(),
        "poll_seconds": int(args.poll_seconds),
        "max_hours": float(args.max_hours),
        "stage_summaries": {},
        "last_execute_pid": None,
        "finalize_status": "pending",
        "finalize_rc": None,
        "status": "running",
    }

    while datetime.now(timezone.utc) < deadline:
        cycle_state_path = cycle_root / "cycle_state.json"
        execute_pids = _extract_execute_pids(cycle_spec_path)

        if not cycle_state_path.exists() and not execute_pids:
            pid = _start_execute(cycle_spec_path, execute_log)
            state["last_execute_pid"] = pid
            execute_pids = [pid]

        stage_roots = _list_stage_roots(cycle_root)
        for stage_name, stage_root in stage_roots.items():
            done = _summarize_stage_if_ready(stage_root, log_path)
            state["stage_summaries"][stage_name] = {
                "root": str(stage_root),
                "summary_ready": bool(done),
                "checked_at": _utc_now_iso(),
            }

        if cycle_state_path.exists() and not execute_pids:
            finalize_rc = _run_command(
                [
                    "python3",
                    "vllm_plugin/experiments/run_iteration_cycle.py",
                    "finalize",
                    "--cycle-spec",
                    str(cycle_spec_path),
                ],
                log_path,
            )
            state["finalize_rc"] = finalize_rc
            state["finalize_status"] = "ok" if finalize_rc == 0 else "error"
            state["status"] = "completed" if finalize_rc == 0 else "failed"
            state["finished_at"] = _utc_now_iso()
            _write_json(state_path, state)
            return

        state["last_polled_at"] = _utc_now_iso()
        state["execute_pids"] = execute_pids
        _write_json(state_path, state)
        time.sleep(max(1, int(args.poll_seconds)))

    state["status"] = "timed_out"
    state["finished_at"] = _utc_now_iso()
    _write_json(state_path, state)


if __name__ == "__main__":
    main()
