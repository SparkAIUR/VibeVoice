#!/usr/bin/env python3
"""
Summarize a completed expert review template into machine-readable scoreboards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from vllm_plugin.experiments.expert_review import summarize_review_payload, write_review_summary
except ModuleNotFoundError:
    from expert_review import summarize_review_payload, write_review_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize expert transcript review artifacts.")
    parser.add_argument(
        "--review-json",
        required=True,
        help="Path to review_template.json after scores have been filled in.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to the review JSON directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    review_json_path = Path(args.review_json).resolve()
    if not review_json_path.exists():
        raise FileNotFoundError(f"Review JSON not found: {review_json_path}")

    payload = json.loads(review_json_path.read_text(encoding="utf-8"))
    summary = summarize_review_payload(payload)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else review_json_path.parent
    outputs = write_review_summary(output_dir, summary)

    print("[INFO] Expert review summary complete.")
    for name, path in outputs.items():
        print(f"[INFO] {name}: {path}")


if __name__ == "__main__":
    main()
