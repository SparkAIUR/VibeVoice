#!/usr/bin/env python3
"""
Run long-audio chunking experiments for VibeVoice ASR.

This script evaluates chunked transcription quality against a full-length
"gold" transcription and captures throughput metrics.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import dataclasses
import difflib
import json
import math
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    from vllm_plugin.experiments.expert_review import make_review_item, write_review_bundle
except ModuleNotFoundError:
    from expert_review import make_review_item, write_review_bundle


SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output "
    "in JSON format."
)


@dataclasses.dataclass
class ManifestItem:
    file_id: str
    path: str
    language: str | None = None
    hotwords: str | None = None


@dataclasses.dataclass
class ApiCallResult:
    success: bool
    status_code: int | None
    raw_text: str
    segments: list[dict[str, Any]]
    latency_sec: float
    retry_count: int
    error: str | None = None
    parse_warning: str | None = None
    usage: dict[str, Any] | None = None


@dataclasses.dataclass
class ScenarioConfig:
    name: str
    chunk_minutes: int | None
    overlap_seconds: float
    context_carry: bool


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return slug or "item"


def _guess_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "video/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
    }
    return mime_map.get(ext, "application/octet-stream")


def _get_duration_seconds_ffprobe(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
    return float(out)


def _read_manifest(manifest_path: Path) -> list[ManifestItem]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data["files"] if isinstance(data, dict) and "files" in data else data
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a list or object with 'files' list.")

    parsed: list[ManifestItem] = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            parsed.append(ManifestItem(file_id=f"file_{i+1:02d}", path=entry))
            continue
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"Invalid manifest entry at index {i}: {entry}")
        item_id = str(entry.get("id") or entry.get("file_id") or f"file_{i+1:02d}")
        parsed.append(
            ManifestItem(
                file_id=item_id,
                path=str(entry["path"]),
                language=entry.get("language"),
                hotwords=entry.get("hotwords"),
            )
        )
    return parsed


def _build_prompt(
    duration_sec: float,
    language: str | None,
    hotwords: str | None,
    context_tail: str | None,
) -> str:
    show_keys = "Start, End, Speaker, Content"
    max_segments = max(12, min(240, int(math.ceil(max(duration_sec, 1.0) / 25.0))))
    parts = [f"This is a {duration_sec:.2f} seconds audio chunk."]
    if language:
        parts.append(f"Language hint: {language}.")
    if hotwords:
        parts.append(f"Extra info (hotwords): {hotwords.strip()}")
    if context_tail:
        parts.append(
            "Previous transcript tail context (for continuity only, do not repeat it):\n"
            f"{context_tail.strip()}"
        )
    parts.append(
        "Use coarse, phrase-level timestamping only (not word-level). "
        "Prefer segment lengths around 20-45 seconds when possible and merge adjacent "
        f"same-speaker spans. Keep the total segment count <= {max_segments}."
    )
    parts.append(
        "Please transcribe it as a strict JSON array with keys: "
        f"{show_keys}. Return JSON only."
    )
    return "\n\n".join(parts)


def _extract_response_content(payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    choices = payload.get("choices") or []
    if not choices:
        return "", payload.get("usage")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        content = "".join(text_parts)
    return str(content), payload.get("usage")


def _normalize_segment(segment: dict[str, Any]) -> dict[str, Any] | None:
    start_keys = ["Start", "start", "Start time", "start_time", "startTime"]
    end_keys = ["End", "end", "End time", "end_time", "endTime"]
    speaker_keys = ["Speaker", "speaker", "Speaker ID", "speaker_id"]
    content_keys = ["Content", "content", "text", "Text"]

    def _first(keys: list[str]) -> Any:
        for key in keys:
            if key in segment:
                return segment[key]
        return None

    start = _first(start_keys)
    end = _first(end_keys)
    content = _first(content_keys)
    speaker = _first(speaker_keys)

    try:
        start_f = float(start)
        end_f = float(end)
    except (TypeError, ValueError):
        return None
    if content is None:
        return None

    return {
        "Start": round(start_f, 3),
        "End": round(end_f, 3),
        "Speaker": speaker,
        "Content": str(content).strip(),
    }


def _strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = [text]
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        candidates.append(array_match.group(0))
    object_match = re.search(r"\{[\s\S]*\}", text)
    if object_match:
        candidates.append(object_match.group(0))
    return candidates


def _recover_segments_from_objects(text: str) -> tuple[list[dict[str, Any]], str | None]:
    recovered: list[dict[str, Any]] = []
    for match in re.finditer(r"\{[^{}]+\}", text):
        chunk = match.group(0)
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        normalized = _normalize_segment(parsed)
        if normalized:
            recovered.append(normalized)

    if recovered:
        return recovered, f"Recovered {len(recovered)} segments from partial JSON."
    return [], "Unable to parse JSON segments."


def _parse_segments(raw_text: str) -> tuple[list[dict[str, Any]], str | None]:
    text = _strip_code_fences(raw_text)
    for candidate in _extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        items: list[Any]
        if isinstance(parsed, dict) and "segments" in parsed and isinstance(parsed["segments"], list):
            items = parsed["segments"]
        elif isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = [parsed]
        else:
            continue

        normalized = [_normalize_segment(item) for item in items if isinstance(item, dict)]
        segments = [item for item in normalized if item is not None]
        if segments:
            return segments, None
    return _recover_segments_from_objects(text)


def _post_transcription_request(
    api_url: str,
    model: str,
    mime: str,
    audio_bytes: bytes,
    prompt_text: str,
    max_tokens: int,
    timeout_seconds: int,
    max_retries: int,
) -> ApiCallResult:
    url = f"{api_url.rstrip('/')}/v1/chat/completions"
    data_url = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('utf-8')}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }

    started = time.perf_counter()
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout_seconds)
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                continue

            body = response.json()
            raw_text, usage = _extract_response_content(body)
            segments, parse_warning = _parse_segments(raw_text)
            elapsed = time.perf_counter() - started
            if not segments:
                last_error = parse_warning or "No segments parsed from response."
                continue

            return ApiCallResult(
                success=True,
                status_code=response.status_code,
                raw_text=raw_text,
                segments=segments,
                latency_sec=elapsed,
                retry_count=attempt,
                parse_warning=parse_warning,
                usage=usage,
            )
        except requests.RequestException as exc:
            last_error = str(exc)

    elapsed = time.perf_counter() - started
    return ApiCallResult(
        success=False,
        status_code=None,
        raw_text="",
        segments=[],
        latency_sec=elapsed,
        retry_count=max_retries,
        error=last_error or "Unknown request error.",
    )


def _extract_chunk_wav_bytes(audio_path: str, start_sec: float, duration_sec: float) -> bytes:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration_sec:.3f}",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-f",
        "wav",
        "-",
    ]
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT)


def _build_chunks(total_duration: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    start = 0.0
    step = max(1.0, chunk_seconds - overlap_seconds)
    while start < total_duration - 1e-6:
        end = min(total_duration, start + chunk_seconds)
        chunks.append((round(start, 3), round(end, 3)))
        if end >= total_duration:
            break
        start += step
    return chunks


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def _join_segment_content(segments: list[dict[str, Any]]) -> str:
    return " ".join(seg.get("Content", "").strip() for seg in segments if seg.get("Content", "").strip())


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_normalize_text(a), b=_normalize_text(b)).ratio()


def _dedupe_overlap_segments(
    segments: list[dict[str, Any]], overlap_seconds: float
) -> tuple[list[dict[str, Any]], int]:
    if not segments:
        return [], 0

    deduped: list[dict[str, Any]] = [segments[0]]
    dropped = 0
    for current in segments[1:]:
        prev = deduped[-1]
        start_gap = abs(float(current["Start"]) - float(prev["Start"]))
        if start_gap <= overlap_seconds and _similarity(current["Content"], prev["Content"]) >= 0.92:
            dropped += 1
            continue
        if float(current["Start"]) < float(prev["End"]) and _similarity(current["Content"], prev["Content"]) >= 0.90:
            dropped += 1
            continue
        deduped.append(current)
    return deduped, dropped


def _offset_segments(segments: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "Start": round(float(seg["Start"]) + offset_sec, 3),
                "End": round(float(seg["End"]) + offset_sec, 3),
                "Speaker": seg.get("Speaker"),
                "Content": seg.get("Content", ""),
            }
        )
    return out


def _compare_to_gold(
    gold_segments: list[dict[str, Any]],
    cand_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_text = _join_segment_content(gold_segments)
    cand_text = _join_segment_content(cand_segments)

    char_ratio = difflib.SequenceMatcher(a=_normalize_text(gold_text), b=_normalize_text(cand_text)).ratio()
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_gold_max_tokens(args: argparse.Namespace) -> int:
    if args.max_tokens_gold is not None:
        return int(args.max_tokens_gold)
    return int(args.max_tokens)


def _resolve_chunk_max_tokens(args: argparse.Namespace) -> int:
    if args.max_tokens_chunk is not None:
        return int(args.max_tokens_chunk)
    return int(args.max_tokens)


def _run_full_transcription(
    item: ManifestItem,
    args: argparse.Namespace,
) -> tuple[ApiCallResult, float]:
    duration = _get_duration_seconds_ffprobe(item.path)
    prompt = _build_prompt(duration, item.language, item.hotwords, context_tail=None)
    audio_bytes = Path(item.path).read_bytes()
    mime = _guess_mime_type(item.path)
    result = _post_transcription_request(
        api_url=args.api_url,
        model=args.model,
        mime=mime,
        audio_bytes=audio_bytes,
        prompt_text=prompt,
        max_tokens=_resolve_gold_max_tokens(args),
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    return result, duration


def _run_chunked_transcription(
    item: ManifestItem,
    scenario: ScenarioConfig,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool, str | None]:
    if scenario.chunk_minutes is None:
        raise ValueError("Chunked transcription requires chunk_minutes.")

    total_duration = _get_duration_seconds_ffprobe(item.path)
    chunk_seconds = float(scenario.chunk_minutes * 60)
    chunks = _build_chunks(total_duration, chunk_seconds, scenario.overlap_seconds)
    merged_segments: list[dict[str, Any]] = []
    chunk_latencies: list[float] = []
    failure_count = 0
    retries = 0
    parse_warnings: list[str] = []
    context_tail: str | None = None

    for idx, (start_sec, end_sec) in enumerate(chunks):
        chunk_duration = end_sec - start_sec
        chunk_bytes = _extract_chunk_wav_bytes(item.path, start_sec, chunk_duration)
        prompt = _build_prompt(
            duration_sec=chunk_duration,
            language=item.language,
            hotwords=item.hotwords,
            context_tail=context_tail if scenario.context_carry else None,
        )

        call = _post_transcription_request(
            api_url=args.api_url,
            model=args.model,
            mime="audio/wav",
            audio_bytes=chunk_bytes,
            prompt_text=prompt,
            max_tokens=_resolve_chunk_max_tokens(args),
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        chunk_latencies.append(call.latency_sec)
        retries += call.retry_count

        if not call.success:
            failure_count += 1
            continue
        if call.parse_warning:
            parse_warnings.append(f"chunk_{idx:03d}: {call.parse_warning}")

        chunk_segments = _offset_segments(call.segments, offset_sec=start_sec)
        merged_segments.extend(chunk_segments)

        if scenario.context_carry:
            merged_text = _join_segment_content(merged_segments)
            if merged_text:
                context_tail = merged_text[-args.context_tail_chars :]

    merged_segments.sort(key=lambda seg: (float(seg["Start"]), float(seg["End"])))
    deduped_segments, dropped = _dedupe_overlap_segments(
        merged_segments, overlap_seconds=scenario.overlap_seconds
    )

    p95_latency = None
    if chunk_latencies:
        sorted_lat = sorted(chunk_latencies)
        idx = min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1)
        p95_latency = sorted_lat[idx]

    details = {
        "chunk_count": len(chunks),
        "chunk_latencies_sec": [round(v, 6) for v in chunk_latencies],
        "chunk_latency_p95_sec": None if p95_latency is None else round(p95_latency, 6),
        "chunk_failures": failure_count,
        "retry_count": retries,
        "overlap_dropped_segments": dropped,
        "parse_warnings": parse_warnings,
        "total_duration_sec": round(total_duration, 6),
    }
    success = bool(deduped_segments) and failure_count == 0
    error = None if success else "One or more chunk requests failed or no segments produced."
    return deduped_segments, details, success, error


def _write_transcript_artifacts(
    output_dir: Path,
    file_id: str,
    scenario_name: str,
    segments: list[dict[str, Any]],
    raw_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    transcript_dir = output_dir / "transcripts" / _slugify(file_id)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    json_path = transcript_dir / f"{_slugify(scenario_name)}.json"
    txt_path = transcript_dir / f"{_slugify(scenario_name)}.txt"

    payload = {"segments": segments}
    if raw_text is not None:
        payload["raw_text"] = raw_text
    if metadata is not None:
        payload["metadata"] = metadata

    _write_json(json_path, payload)
    _write_text(txt_path, _join_segment_content(segments))


def _summarize_runs_to_csv(summary_csv: Path, runs: list[dict[str, Any]]) -> None:
    if not runs:
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for run in runs for k in run.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(run)


def _summarize_runs_to_markdown(
    summary_md: Path,
    runs: list[dict[str, Any]],
    threshold: float,
) -> str:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_scenario.setdefault(run["scenario"], []).append(run)

    lines: list[str] = []
    lines.append("# Chunking Experiment Summary")
    lines.append("")
    lines.append(f"- Generated: {_utc_now_iso()}")
    lines.append(f"- Quality threshold (word drift): {threshold:.4f}")
    lines.append("")
    lines.append("| Scenario | Runs | Success | Mean Word Drift | Mean Char Drift | Mean RTF |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for scenario in sorted(by_scenario):
        rows = by_scenario[scenario]
        success_rows = [row for row in rows if row.get("success")]
        word_vals = [row["word_drift"] for row in success_rows if row.get("word_drift") is not None]
        char_vals = [row["char_drift"] for row in success_rows if row.get("char_drift") is not None]
        rtf_vals = [row["rtf"] for row in success_rows if row.get("rtf") is not None]
        lines.append(
            "| {scenario} | {runs} | {ok} | {word} | {char} | {rtf} |".format(
                scenario=scenario,
                runs=len(rows),
                ok=len(success_rows),
                word=f"{statistics.mean(word_vals):.4f}" if word_vals else "NA",
                char=f"{statistics.mean(char_vals):.4f}" if char_vals else "NA",
                rtf=f"{statistics.mean(rtf_vals):.4f}" if rtf_vals else "NA",
            )
        )

    text = "\n".join(lines) + "\n"
    _write_text(summary_md, text)
    return text


def _build_manual_review(
    manual_review_path: Path,
    runs: list[dict[str, Any]],
    threshold: float,
    transcript_root: Path,
) -> None:
    lines: list[str] = ["# Manual Diff Review", ""]
    flagged = [
        run
        for run in runs
        if run.get("scenario") != "gold_full"
        and run.get("success")
        and run.get("word_drift") is not None
        and (run["word_drift"] > threshold or run["word_drift"] > 0.01)
    ]

    if not flagged:
        lines.append("No scenarios exceeded review thresholds.")
        _write_text(manual_review_path, "\n".join(lines) + "\n")
        return

    for run in flagged:
        file_slug = _slugify(str(run["file_id"]))
        scenario_slug = _slugify(str(run["scenario"]))
        gold_json_path = transcript_root / file_slug / "gold_full.json"
        cand_json_path = transcript_root / file_slug / f"{scenario_slug}.json"
        gold_payload = json.loads(gold_json_path.read_text(encoding="utf-8"))
        cand_payload = json.loads(cand_json_path.read_text(encoding="utf-8"))
        gold_text = _join_segment_content(gold_payload.get("segments", []))
        cand_text = _join_segment_content(cand_payload.get("segments", []))
        snippets = _extract_diff_snippets(gold_text, cand_text)
        lines.append(f"## {run['file_id']} - {run['scenario']}")
        lines.append(f"- Word drift: {run['word_drift']:.4f}")
        lines.append(f"- Char drift: {run['char_drift']:.4f}")
        if snippets:
            lines.extend(snippets)
        else:
            lines.append("- No diff snippets could be extracted.")
        lines.append("")
    _write_text(manual_review_path, "\n".join(lines) + "\n")


def _build_expert_review_bundle(
    output_dir: Path,
    runs: list[dict[str, Any]],
    transcript_root: Path,
) -> None:
    items: list[dict[str, Any]] = []
    for run in runs:
        if run.get("scenario") == "gold_full":
            continue
        if run.get("file_id") == "__aggregate__":
            continue
        if not run.get("success"):
            continue
        file_slug = _slugify(str(run["file_id"]))
        scenario_slug = _slugify(str(run["scenario"]))
        gold_json_path = transcript_root / file_slug / "gold_full.json"
        cand_json_path = transcript_root / file_slug / f"{scenario_slug}.json"
        gold_txt_path = transcript_root / file_slug / "gold_full.txt"
        cand_txt_path = transcript_root / file_slug / f"{scenario_slug}.txt"
        if not gold_json_path.exists() or not cand_json_path.exists():
            continue

        gold_payload = json.loads(gold_json_path.read_text(encoding="utf-8"))
        cand_payload = json.loads(cand_json_path.read_text(encoding="utf-8"))
        diff_snippets = _extract_diff_snippets(
            _join_segment_content(gold_payload.get("segments", [])),
            _join_segment_content(cand_payload.get("segments", [])),
        )
        items.append(
            make_review_item(
                comparison={
                    "file_id": run.get("file_id"),
                    "scenario": run.get("scenario"),
                },
                gold_txt_path=gold_txt_path,
                candidate_txt_path=cand_txt_path,
                metrics={
                    key: run.get(key)
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
                diff_snippets=diff_snippets,
            )
        )

    write_review_bundle(
        output_dir=output_dir,
        title="Chunking Experiment Expert Review Template",
        items=items,
        group_by_fields=("scenario",),
        source_kind="chunking_experiment",
        metadata={
            "transcript_root": str(transcript_root),
            "review_scope": "All successful non-gold transcripts in this run.",
        },
    )


def _iter_quality_scenarios(
    chunk_minutes: Iterable[int],
    overlap_seconds: float,
    scenario_modes: Iterable[str],
) -> list[ScenarioConfig]:
    mode_set = {str(mode).strip().lower() for mode in scenario_modes}
    scenarios: list[ScenarioConfig] = []
    for minutes in chunk_minutes:
        if "no_overlap" in mode_set:
            scenarios.append(
                ScenarioConfig(
                    name=f"chunk_{minutes}m_no_overlap",
                    chunk_minutes=minutes,
                    overlap_seconds=0.0,
                    context_carry=False,
                )
            )
        if "overlap_context" in mode_set:
            scenarios.append(
                ScenarioConfig(
                    name=f"chunk_{minutes}m_overlap_context",
                    chunk_minutes=minutes,
                    overlap_seconds=overlap_seconds,
                    context_carry=True,
                )
            )
    return scenarios


def _pick_best_chunk_scenario(runs: list[dict[str, Any]], threshold: float) -> str | None:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        scenario = str(run["scenario"])
        if scenario == "gold_full":
            continue
        if not run.get("success"):
            continue
        if run.get("word_drift") is None:
            continue
        by_scenario.setdefault(scenario, []).append(run)

    if not by_scenario:
        return None

    ordered = sorted(
        by_scenario.items(),
        key=lambda kv: statistics.mean(float(item["word_drift"]) for item in kv[1]),
    )
    passing = [
        (name, rows)
        for name, rows in ordered
        if statistics.mean(float(item["word_drift"]) for item in rows) <= threshold
    ]
    selected = passing[0] if passing else ordered[0]
    return selected[0]


def _find_scenario_by_name(name: str, scenarios: list[ScenarioConfig]) -> ScenarioConfig | None:
    for scenario in scenarios:
        if scenario.name == name:
            return scenario
    return None


def _run_single_scenario_for_item(
    item: ManifestItem,
    scenario: ScenarioConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    segments, details, success, error = _run_chunked_transcription(item, scenario, args)
    elapsed = time.perf_counter() - started
    duration = details.get("total_duration_sec") or _get_duration_seconds_ffprobe(item.path)
    return {
        "file_id": item.file_id,
        "file_path": item.path,
        "scenario": scenario.name,
        "success": success,
        "error": error,
        "duration_sec": round(float(duration), 6),
        "wall_time_sec": round(elapsed, 6),
        "rtf": round(elapsed / float(duration), 6) if duration else None,
        "chunks_processed_per_min": round(
            (details["chunk_count"] / elapsed) * 60.0, 6
        )
        if elapsed > 0
        else None,
        "chunk_count": details["chunk_count"],
        "chunk_failures": details["chunk_failures"],
        "retry_count": details["retry_count"],
        "chunk_latency_p95_sec": details["chunk_latency_p95_sec"],
        "segments": segments,
        "details": details,
    }


def _run_throughput_matrix(
    items: list[ManifestItem],
    scenario: ScenarioConfig,
    worker_counts: list[int],
    args: argparse.Namespace,
    output_dir: Path,
    runs_jsonl_path: Path,
    collected_runs: list[dict[str, Any]],
) -> None:
    for workers in worker_counts:
        wave_start = time.perf_counter()
        wave_results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_single_scenario_for_item, item=item, scenario=scenario, args=args)
                for item in items
            ]
            for fut in concurrent.futures.as_completed(futures):
                wave_results.append(fut.result())

        elapsed = time.perf_counter() - wave_start
        success_count = sum(1 for row in wave_results if row["success"])
        total_duration = sum(float(row["duration_sec"]) for row in wave_results)
        throughput_summary = {
            "file_id": "__aggregate__",
            "file_path": "",
            "scenario": f"throughput_workers_{workers}_{scenario.name}",
            "success": success_count == len(wave_results),
            "error": None,
            "duration_sec": round(total_duration, 6),
            "wall_time_sec": round(elapsed, 6),
            "rtf": round(elapsed / total_duration, 6) if total_duration else None,
            "chunks_processed_per_min": None,
            "chunk_count": sum(int(row["chunk_count"]) for row in wave_results),
            "chunk_failures": sum(int(row["chunk_failures"]) for row in wave_results),
            "retry_count": sum(int(row["retry_count"]) for row in wave_results),
            "chunk_latency_p95_sec": None,
            "word_drift": None,
            "char_drift": None,
            "segment_count_delta": None,
            "boundary_mae_sec": None,
            "quality_pass": None,
            "throughput_files_per_hour": round((len(wave_results) / elapsed) * 3600.0, 6)
            if elapsed > 0
            else None,
            "throughput_audio_hours_per_hour": round(
                (total_duration / 3600.0) / (elapsed / 3600.0), 6
            )
            if elapsed > 0
            else None,
            "worker_count": workers,
            "generated_at": _utc_now_iso(),
        }

        for row in wave_results:
            file_run = {
                "file_id": row["file_id"],
                "file_path": row["file_path"],
                "scenario": f"throughput_workers_{workers}_{scenario.name}",
                "success": row["success"],
                "error": row["error"],
                "duration_sec": row["duration_sec"],
                "wall_time_sec": row["wall_time_sec"],
                "rtf": row["rtf"],
                "chunks_processed_per_min": row["chunks_processed_per_min"],
                "chunk_count": row["chunk_count"],
                "chunk_failures": row["chunk_failures"],
                "retry_count": row["retry_count"],
                "chunk_latency_p95_sec": row["chunk_latency_p95_sec"],
                "word_drift": None,
                "char_drift": None,
                "segment_count_delta": None,
                "boundary_mae_sec": None,
                "quality_pass": None,
                "throughput_files_per_hour": None,
                "throughput_audio_hours_per_hour": None,
                "worker_count": workers,
                "generated_at": _utc_now_iso(),
            }
            collected_runs.append(file_run)
            with runs_jsonl_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(file_run, ensure_ascii=True) + "\n")

            _write_transcript_artifacts(
                output_dir=output_dir,
                file_id=row["file_id"],
                scenario_name=file_run["scenario"],
                segments=row["segments"],
                metadata={"details": row["details"], "worker_count": workers},
            )

        collected_runs.append(throughput_summary)
        with runs_jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(throughput_summary, ensure_ascii=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunking quality and throughput experiments for VibeVoice ASR."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON manifest. Format: list of paths or {'files':[...]} entries.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/chunking_experiment",
        help="Directory where experiment artifacts are written.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="vLLM API URL.",
    )
    parser.add_argument(
        "--model",
        default="vibevoice",
        help="Served model name.",
    )
    parser.add_argument(
        "--chunk-minutes",
        nargs="+",
        type=int,
        default=[10, 20, 30],
        help="Chunk sizes (minutes) to evaluate.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=30.0,
        help="Overlap size in seconds for overlap+context scenarios.",
    )
    parser.add_argument(
        "--context-tail-chars",
        type=int,
        default=800,
        help="How many chars of prior transcript to carry as context.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="max_tokens for each API request.",
    )
    parser.add_argument(
        "--max-tokens-gold",
        type=int,
        default=None,
        help="Override max_tokens for full-length gold runs only.",
    )
    parser.add_argument(
        "--max-tokens-chunk",
        type=int,
        default=None,
        help="Override max_tokens for chunked runs only.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for each API call on parse/request failure.",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.02,
        help="Word drift threshold for quality pass/fail.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of manifest files to run.",
    )
    parser.add_argument(
        "--run-throughput",
        action="store_true",
        help="Run throughput matrix after quality run.",
    )
    parser.add_argument(
        "--throughput-workers",
        nargs="+",
        type=int,
        default=[1, 2, 4, 6],
        help="Worker counts for throughput matrix.",
    )
    parser.add_argument(
        "--throughput-scenario",
        default=None,
        help="Override scenario for throughput (defaults to best quality chunked scenario).",
    )
    parser.add_argument(
        "--scenario-modes",
        nargs="+",
        choices=["no_overlap", "overlap_context"],
        default=["no_overlap", "overlap_context"],
        help="Chunking scenario families to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    items = _read_manifest(manifest_path)
    if args.max_files is not None:
        items = items[: args.max_files]
    if not items:
        raise ValueError("No files found in manifest after filtering.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_jsonl_path = output_dir / "runs.jsonl"
    summary_csv_path = output_dir / "summary.csv"
    summary_md_path = output_dir / "summary.md"
    manual_review_path = output_dir / "manual_review.md"

    _write_json(
        output_dir / "manifest_used.json",
        {
            "generated_at": _utc_now_iso(),
            "manifest_source": str(manifest_path),
            "files": [dataclasses.asdict(item) for item in items],
        },
    )

    quality_scenarios = _iter_quality_scenarios(
        args.chunk_minutes,
        args.overlap_seconds,
        args.scenario_modes,
    )
    all_runs: list[dict[str, Any]] = []

    for item in items:
        print(f"[INFO] Running gold transcription for {item.file_id} :: {item.path}")
        gold_started = time.perf_counter()
        gold_result, duration = _run_full_transcription(item, args)
        gold_elapsed = time.perf_counter() - gold_started

        gold_row = {
            "file_id": item.file_id,
            "file_path": item.path,
            "scenario": "gold_full",
            "success": gold_result.success,
            "error": gold_result.error,
            "duration_sec": round(duration, 6),
            "wall_time_sec": round(gold_elapsed, 6),
            "rtf": round(gold_elapsed / duration, 6) if duration else None,
            "chunks_processed_per_min": round((1.0 / gold_elapsed) * 60.0, 6)
            if gold_elapsed > 0
            else None,
            "chunk_count": 1,
            "chunk_failures": 0 if gold_result.success else 1,
            "retry_count": gold_result.retry_count,
            "chunk_latency_p95_sec": round(gold_result.latency_sec, 6),
            "word_drift": 0.0 if gold_result.success else None,
            "char_drift": 0.0 if gold_result.success else None,
            "segment_count_delta": 0 if gold_result.success else None,
            "boundary_mae_sec": 0.0 if gold_result.success else None,
            "quality_pass": True if gold_result.success else None,
            "throughput_files_per_hour": None,
            "throughput_audio_hours_per_hour": None,
            "worker_count": None,
            "generated_at": _utc_now_iso(),
        }
        all_runs.append(gold_row)
        with runs_jsonl_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(gold_row, ensure_ascii=True) + "\n")

        _write_transcript_artifacts(
            output_dir=output_dir,
            file_id=item.file_id,
            scenario_name="gold_full",
            segments=gold_result.segments,
            raw_text=gold_result.raw_text,
            metadata={
                "duration_sec": duration,
                "api_latency_sec": gold_result.latency_sec,
                "retry_count": gold_result.retry_count,
                "parse_warning": gold_result.parse_warning,
                "usage": gold_result.usage,
            },
        )

        if not gold_result.success:
            print(f"[WARN] Gold run failed for {item.file_id}: {gold_result.error}")
            continue

        for scenario in quality_scenarios:
            print(f"[INFO] Running {scenario.name} for {item.file_id}")
            run_started = time.perf_counter()
            segments, details, success, error = _run_chunked_transcription(item, scenario, args)
            elapsed = time.perf_counter() - run_started
            comparison = _compare_to_gold(gold_result.segments, segments) if success else {}
            row = {
                "file_id": item.file_id,
                "file_path": item.path,
                "scenario": scenario.name,
                "success": success,
                "error": error,
                "duration_sec": round(details["total_duration_sec"], 6),
                "wall_time_sec": round(elapsed, 6),
                "rtf": round(elapsed / details["total_duration_sec"], 6)
                if details["total_duration_sec"]
                else None,
                "chunks_processed_per_min": round(
                    (details["chunk_count"] / elapsed) * 60.0, 6
                )
                if elapsed > 0
                else None,
                "chunk_count": details["chunk_count"],
                "chunk_failures": details["chunk_failures"],
                "retry_count": details["retry_count"],
                "chunk_latency_p95_sec": details["chunk_latency_p95_sec"],
                "word_drift": comparison.get("word_drift"),
                "char_drift": comparison.get("char_drift"),
                "segment_count_delta": comparison.get("segment_count_delta"),
                "boundary_mae_sec": comparison.get("boundary_mae_sec"),
                "quality_pass": (
                    comparison.get("word_drift") is not None
                    and comparison["word_drift"] <= args.quality_threshold
                )
                if success
                else None,
                "throughput_files_per_hour": None,
                "throughput_audio_hours_per_hour": None,
                "worker_count": None,
                "generated_at": _utc_now_iso(),
            }
            all_runs.append(row)
            with runs_jsonl_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=True) + "\n")

            _write_transcript_artifacts(
                output_dir=output_dir,
                file_id=item.file_id,
                scenario_name=scenario.name,
                segments=segments,
                metadata={"details": details},
            )

    selected_throughput_scenario = args.throughput_scenario or _pick_best_chunk_scenario(
        all_runs, args.quality_threshold
    )
    if args.run_throughput and selected_throughput_scenario:
        scenario_obj = _find_scenario_by_name(selected_throughput_scenario, quality_scenarios)
        if scenario_obj is None:
            raise ValueError(
                f"Throughput scenario '{selected_throughput_scenario}' not found in quality scenarios."
            )
        print(f"[INFO] Running throughput matrix using scenario: {selected_throughput_scenario}")
        _run_throughput_matrix(
            items=items,
            scenario=scenario_obj,
            worker_counts=args.throughput_workers,
            args=args,
            output_dir=output_dir,
            runs_jsonl_path=runs_jsonl_path,
            collected_runs=all_runs,
        )
    elif args.run_throughput and not selected_throughput_scenario:
        print("[WARN] Skipping throughput matrix because no successful chunk scenario was found.")

    _summarize_runs_to_csv(summary_csv_path, all_runs)
    summary_text = _summarize_runs_to_markdown(summary_md_path, all_runs, args.quality_threshold)
    _build_manual_review(
        manual_review_path=manual_review_path,
        runs=all_runs,
        threshold=args.quality_threshold,
        transcript_root=output_dir / "transcripts",
    )
    _build_expert_review_bundle(
        output_dir=output_dir,
        runs=all_runs,
        transcript_root=output_dir / "transcripts",
    )

    print("[INFO] Experiment complete.")
    print(f"[INFO] Artifacts directory: {output_dir}")
    print("[INFO] Summary:")
    print(summary_text)


if __name__ == "__main__":
    main()
