#!/usr/bin/env python3
"""
Run long-audio chunking experiments with variant stitching strategies.

This script extends the baseline chunking harness with idea-variant hooks:
- sentence-aware seam selection
- seam micro-redo
- shifted-grid consensus
- silence-aligned boundaries
- dynamic lexicon hotwords
"""

from __future__ import annotations

import argparse
import base64
import collections
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


SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output "
    "in JSON format."
)

RAW_VARIANT_FEATURES = {
    "sentence_seam",
    "seam_micro_redo",
    "shifted_consensus",
    "silence_aligned_boundaries",
    "dynamic_lexicon",
}

VARIANT_ALIASES: dict[str, set[str]] = {
    "baseline_overlap_context": set(),
    "idea1_sentence_seam": {"sentence_seam"},
    "idea2_seam_micro_redo": {"seam_micro_redo"},
    "idea3_shifted_consensus": {"shifted_consensus"},
    "idea4_silence_aligned_boundaries": {"silence_aligned_boundaries"},
    "idea5_dynamic_lexicon": {"dynamic_lexicon"},
}

STOPWORDS_EN = {
    "the",
    "and",
    "that",
    "have",
    "for",
    "you",
    "with",
    "this",
    "from",
    "your",
    "about",
    "what",
    "when",
    "where",
    "there",
    "they",
    "their",
    "would",
    "could",
    "should",
    "been",
    "were",
    "will",
    "into",
    "because",
    "just",
    "very",
    "than",
    "then",
    "them",
    "over",
    "under",
    "also",
}

STOPWORDS_ES = {
    "que",
    "para",
    "como",
    "pero",
    "porque",
    "desde",
    "hasta",
    "sobre",
    "entre",
    "donde",
    "cuando",
    "esta",
    "este",
    "esto",
    "tiene",
    "tengo",
    "usted",
    "ustedes",
    "ellos",
    "ellas",
    "nosotros",
    "nosotras",
    "tambien",
    "solo",
    "muy",
    "bien",
    "pues",
    "entonces",
    "mismo",
    "misma",
    "antes",
    "despues",
    "aqui",
    "alla",
    "algo",
    "nada",
    "cada",
}


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


def _resolve_variant_features(variant_spec: str) -> set[str]:
    tokens = [t.strip() for t in str(variant_spec).split(",") if t.strip()]
    if not tokens:
        return set()
    resolved: set[str] = set()
    for token in tokens:
        if token in VARIANT_ALIASES:
            resolved.update(VARIANT_ALIASES[token])
            continue
        if token in RAW_VARIANT_FEATURES:
            resolved.add(token)
            continue
        raise ValueError(
            f"Unknown variant token '{token}'. "
            f"Known aliases={sorted(VARIANT_ALIASES)} features={sorted(RAW_VARIANT_FEATURES)}"
        )
    return resolved


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


def _parse_hotwords(text: str | None) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return out


def _merge_hotwords(base_hotwords: str | None, dynamic_terms: list[str]) -> str | None:
    merged: list[str] = []
    seen: set[str] = set()
    for token in _parse_hotwords(base_hotwords) + dynamic_terms:
        norm = token.strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(norm)
    if not merged:
        return None
    return ", ".join(merged)


def _tokenize_terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'-]*", text.lower())


def _stopwords_for_language(language: str | None) -> set[str]:
    lang = (language or "").strip().lower()
    if "spanish" in lang or lang in {"es", "es-es", "es-mx"}:
        return STOPWORDS_ES
    if "english" in lang or lang in {"en", "en-us", "en-gb"}:
        return STOPWORDS_EN
    return STOPWORDS_EN | STOPWORDS_ES


def _extract_dynamic_terms(
    text: str,
    language: str | None,
    min_len: int,
) -> list[str]:
    stopwords = _stopwords_for_language(language)
    terms: list[str] = []
    for token in _tokenize_terms(text):
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if token.isdigit():
            continue
        terms.append(token)
    return terms


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


def _build_chunk_starts(total_duration: float, chunk_seconds: float, overlap_seconds: float) -> list[float]:
    starts: list[float] = []
    start = 0.0
    step = max(1.0, chunk_seconds - overlap_seconds)
    while start < total_duration - 1e-6:
        starts.append(round(start, 3))
        end = start + chunk_seconds
        if end >= total_duration:
            break
        start += step
    return starts


def _chunks_from_starts(
    starts: list[float],
    total_duration: float,
    chunk_seconds: float,
) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    for start in starts:
        end = min(total_duration, start + chunk_seconds)
        if end - start < 1e-3:
            continue
        chunks.append((round(start, 3), round(end, 3)))
    if not chunks:
        chunks.append((0.0, round(total_duration, 3)))
    return chunks


def _build_chunks(total_duration: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    starts = _build_chunk_starts(total_duration, chunk_seconds, overlap_seconds)
    return _chunks_from_starts(starts, total_duration, chunk_seconds)


def _detect_silence_points_ffmpeg(audio_path: str, noise_db: float, min_duration: float) -> list[float]:
    cmd = [
        "ffmpeg",
        "-v",
        "info",
        "-i",
        audio_path,
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f",
        "null",
        "-",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as exc:
        out = exc.output.decode("utf-8", errors="ignore")
    points: list[float] = []
    for match in re.finditer(r"silence_end:\s*([0-9.]+)", out):
        try:
            points.append(float(match.group(1)))
        except ValueError:
            continue
    return sorted(points)


def _align_starts_to_silence(
    starts: list[float],
    silence_points: list[float],
    search_window_sec: float,
    total_duration: float,
) -> list[float]:
    if not silence_points or len(starts) <= 1:
        return starts
    aligned = [starts[0]]
    min_spacing = 20.0
    for nominal in starts[1:]:
        candidates = [p for p in silence_points if abs(p - nominal) <= search_window_sec]
        target = min(candidates, key=lambda p: abs(p - nominal)) if candidates else nominal
        lower = aligned[-1] + min_spacing
        upper = max(lower, total_duration - 1.0)
        target = max(lower, min(target, upper))
        aligned.append(round(target, 3))
    return aligned


def _build_shifted_chunks(
    total_duration: float,
    chunk_seconds: float,
    overlap_seconds: float,
    shift_seconds: float,
) -> list[tuple[float, float]]:
    step = max(1.0, chunk_seconds - overlap_seconds)
    starts: list[float] = [0.0]
    start = shift_seconds
    while start < total_duration - 1e-6:
        starts.append(round(start, 3))
        end = start + chunk_seconds
        if end >= total_duration:
            break
        start += step
    starts = sorted(set(starts))
    return _chunks_from_starts(starts, total_duration, chunk_seconds)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def _join_segment_content(segments: list[dict[str, Any]]) -> str:
    return " ".join(seg.get("Content", "").strip() for seg in segments if seg.get("Content", "").strip())


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_normalize_text(a), b=_normalize_text(b)).ratio()


def _is_sentence_complete(text: str) -> bool:
    stripped = text.strip()
    return bool(re.search(r"[.!?][\"')\]]*$", stripped) or re.search(r"[.!?]$", stripped))


def _public_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "Start": round(float(seg["Start"]), 3),
                "End": round(float(seg["End"]), 3),
                "Speaker": seg.get("Speaker"),
                "Content": str(seg.get("Content", "")),
            }
        )
    return out


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


def _offset_segments(
    segments: list[dict[str, Any]],
    offset_sec: float,
    chunk_idx: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        out.append(
            {
                "Start": round(float(seg["Start"]) + offset_sec, 3),
                "End": round(float(seg["End"]) + offset_sec, 3),
                "Speaker": seg.get("Speaker"),
                "Content": seg.get("Content", ""),
                "_chunk_idx": chunk_idx,
            }
        )
    return out


def _apply_chunk_cut(
    segments: list[dict[str, Any]],
    next_chunk_idx: int,
    cut_time: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        chunk_idx = int(seg.get("_chunk_idx", 0))
        start = float(seg["Start"])
        end = float(seg["End"])
        if chunk_idx < next_chunk_idx:
            if start >= cut_time:
                continue
            if end > cut_time:
                seg = dict(seg)
                seg["End"] = round(cut_time, 3)
            out.append(seg)
            continue

        if end <= cut_time:
            continue
        if start < cut_time:
            seg = dict(seg)
            seg["Start"] = round(cut_time, 3)
        out.append(seg)
    return out


def _choose_sentence_cut_time(
    segments: list[dict[str, Any]],
    boundary_sec: float,
    next_chunk_idx: int,
    seam_window_sec: float,
) -> float:
    left_candidates: list[float] = []
    right_candidates: list[float] = []
    for seg in segments:
        chunk_idx = int(seg.get("_chunk_idx", 0))
        start = float(seg["Start"])
        end = float(seg["End"])
        content = str(seg.get("Content", ""))
        if chunk_idx < next_chunk_idx:
            if boundary_sec - seam_window_sec <= end <= boundary_sec + seam_window_sec and _is_sentence_complete(content):
                left_candidates.append(end)
        else:
            if boundary_sec - seam_window_sec <= start <= boundary_sec + seam_window_sec and _is_sentence_complete(content):
                right_candidates.append(start)
    if left_candidates:
        return round(max(left_candidates), 3)
    if right_candidates:
        return round(min(right_candidates), 3)
    return round(boundary_sec, 3)


def _pick_sentence_preferred_segment(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_complete = _is_sentence_complete(str(a.get("Content", "")))
    b_complete = _is_sentence_complete(str(b.get("Content", "")))
    if a_complete and not b_complete:
        return dict(a)
    if b_complete and not a_complete:
        return dict(b)
    if len(str(b.get("Content", ""))) > len(str(a.get("Content", ""))):
        return dict(b)
    return dict(a)


def _consensus_merge_segments(
    primary_segments: list[dict[str, Any]],
    shifted_segments: list[dict[str, Any]],
    overlap_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    if not primary_segments:
        return shifted_segments, 0
    if not shifted_segments:
        return primary_segments, 0

    out: list[dict[str, Any]] = []
    used_shift: set[int] = set()
    matched = 0
    max_start_diff = max(15.0, overlap_seconds + 10.0)

    for primary in primary_segments:
        best_idx = -1
        best_score = 0.0
        p_start = float(primary["Start"])
        p_text = str(primary.get("Content", ""))
        for idx, shifted in enumerate(shifted_segments):
            if idx in used_shift:
                continue
            s_start = float(shifted["Start"])
            if abs(p_start - s_start) > max_start_diff:
                continue
            score = _similarity(p_text, str(shifted.get("Content", "")))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= 0.86:
            candidate = _pick_sentence_preferred_segment(primary, shifted_segments[best_idx])
            candidate["Start"] = round(
                (float(primary["Start"]) + float(shifted_segments[best_idx]["Start"])) / 2.0, 3
            )
            candidate["End"] = round(
                (float(primary["End"]) + float(shifted_segments[best_idx]["End"])) / 2.0, 3
            )
            out.append(candidate)
            used_shift.add(best_idx)
            matched += 1
        else:
            out.append(dict(primary))

    out.sort(key=lambda seg: (float(seg["Start"]), float(seg["End"])))
    deduped, dropped = _dedupe_overlap_segments(out, overlap_seconds=overlap_seconds)
    return deduped, matched + dropped


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


def _segments_in_window(segments: list[dict[str, Any]], start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        seg_start = float(seg["Start"])
        seg_end = float(seg["End"])
        if seg_end <= start_sec:
            continue
        if seg_start >= end_sec:
            continue
        out.append(seg)
    return out


def _compute_seam_metrics(
    gold_segments: list[dict[str, Any]],
    cand_segments: list[dict[str, Any]],
    seam_boundaries: list[float],
    seam_window_sec: float,
) -> dict[str, Any]:
    if not seam_boundaries:
        return {"seam_word_drift": None, "seam_boundary_mae_sec": None}
    word_vals: list[float] = []
    boundary_vals: list[float] = []
    for seam in seam_boundaries:
        win_start = max(0.0, seam - seam_window_sec)
        win_end = seam + seam_window_sec
        gold_win = _segments_in_window(gold_segments, win_start, win_end)
        cand_win = _segments_in_window(cand_segments, win_start, win_end)
        if not gold_win or not cand_win:
            continue
        compare = _compare_to_gold(gold_win, cand_win)
        if compare.get("word_drift") is not None:
            word_vals.append(float(compare["word_drift"]))
        if compare.get("boundary_mae_sec") is not None:
            boundary_vals.append(float(compare["boundary_mae_sec"]))
    return {
        "seam_word_drift": round(statistics.mean(word_vals), 6) if word_vals else None,
        "seam_boundary_mae_sec": round(statistics.mean(boundary_vals), 6) if boundary_vals else None,
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


def _load_gold_from_artifact(
    item: ManifestItem,
    gold_artifact_dir: Path,
) -> tuple[ApiCallResult, float]:
    transcript_path = (
        gold_artifact_dir / "transcripts" / _slugify(item.file_id) / "gold_full.json"
    )
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Missing cached gold transcript for {item.file_id}: {transcript_path}"
        )
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments_raw = payload.get("segments", [])
    if not isinstance(segments_raw, list):
        raise ValueError(f"Invalid cached gold payload (segments not list): {transcript_path}")
    segments: list[dict[str, Any]] = []
    for entry in segments_raw:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_segment(entry)
        if normalized:
            segments.append(normalized)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    duration = float(metadata.get("duration_sec")) if metadata.get("duration_sec") else _get_duration_seconds_ffprobe(item.path)
    raw_text = str(payload.get("raw_text") or _join_segment_content(segments))
    result = ApiCallResult(
        success=bool(segments),
        status_code=None,
        raw_text=raw_text,
        segments=segments,
        latency_sec=0.0,
        retry_count=0,
        error=None if segments else "Cached gold transcript is empty.",
        parse_warning=None,
        usage=None,
    )
    return result, duration


def _run_chunks_once(
    item: ManifestItem,
    scenario: ScenarioConfig,
    args: argparse.Namespace,
    chunks: list[tuple[float, float]],
    variant_features: set[str],
    chunk_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged_segments: list[dict[str, Any]] = []
    chunk_latencies: list[float] = []
    failure_count = 0
    retries = 0
    parse_warnings: list[str] = []
    context_tail: str | None = None
    lex_counter: collections.Counter[str] = collections.Counter()

    for idx, (start_sec, end_sec) in enumerate(chunks):
        chunk_duration = end_sec - start_sec
        chunk_bytes = _extract_chunk_wav_bytes(item.path, start_sec, chunk_duration)

        dynamic_terms: list[str] = []
        if "dynamic_lexicon" in variant_features and lex_counter:
            dynamic_terms = [token for token, _ in lex_counter.most_common(args.dynamic_lexicon_terms)]
        hotwords = _merge_hotwords(item.hotwords, dynamic_terms)

        prompt = _build_prompt(
            duration_sec=chunk_duration,
            language=item.language,
            hotwords=hotwords,
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

        chunk_segments = _offset_segments(
            call.segments,
            offset_sec=start_sec,
            chunk_idx=chunk_index_offset + idx,
        )
        merged_segments.extend(chunk_segments)

        if "dynamic_lexicon" in variant_features:
            for token in _extract_dynamic_terms(
                _join_segment_content(call.segments),
                item.language,
                min_len=args.dynamic_lexicon_min_len,
            ):
                lex_counter[token] += 1

        if scenario.context_carry:
            merged_text = _join_segment_content(merged_segments)
            if merged_text:
                context_tail = merged_text[-args.context_tail_chars :]

    seam_boundaries = [chunks[i][0] for i in range(1, len(chunks))]
    seam_cut_points: list[float] = []
    if "sentence_seam" in variant_features and seam_boundaries:
        for seam_idx, seam in enumerate(seam_boundaries, start=1):
            cut = _choose_sentence_cut_time(
                segments=merged_segments,
                boundary_sec=seam,
                next_chunk_idx=chunk_index_offset + seam_idx,
                seam_window_sec=args.seam_window_seconds,
            )
            seam_cut_points.append(cut)
            merged_segments = _apply_chunk_cut(
                segments=merged_segments,
                next_chunk_idx=chunk_index_offset + seam_idx,
                cut_time=cut,
            )

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
        "chunk_ranges_sec": [[round(a, 3), round(b, 3)] for a, b in chunks],
        "seam_boundaries_sec": [round(v, 3) for v in seam_boundaries],
        "seam_cut_points_sec": [round(v, 3) for v in seam_cut_points],
        "chunk_latencies_sec": [round(v, 6) for v in chunk_latencies],
        "chunk_latency_p95_sec": None if p95_latency is None else round(p95_latency, 6),
        "chunk_failures": failure_count,
        "retry_count": retries,
        "overlap_dropped_segments": dropped,
        "parse_warnings": parse_warnings,
    }
    return deduped_segments, details


def _run_seam_micro_redo(
    item: ManifestItem,
    scenario: ScenarioConfig,
    args: argparse.Namespace,
    segments: list[dict[str, Any]],
    seam_boundaries: list[float],
    total_duration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated = list(segments)
    redo_latencies: list[float] = []
    redo_retries = 0
    redo_warnings: list[str] = []
    successful_redos = 0

    for seam in seam_boundaries:
        win_start = max(0.0, seam - args.micro_redo_window_seconds)
        win_end = min(total_duration, seam + args.micro_redo_window_seconds)
        if win_end - win_start < 2.0:
            continue

        chunk_bytes = _extract_chunk_wav_bytes(item.path, win_start, win_end - win_start)
        left_context_segments = [seg for seg in updated if float(seg["End"]) <= win_start]
        context_tail = None
        if scenario.context_carry and left_context_segments:
            left_text = _join_segment_content(left_context_segments)
            context_tail = left_text[-args.context_tail_chars :] if left_text else None

        prompt = _build_prompt(
            duration_sec=win_end - win_start,
            language=item.language,
            hotwords=item.hotwords,
            context_tail=context_tail,
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
        redo_latencies.append(call.latency_sec)
        redo_retries += call.retry_count
        if not call.success:
            redo_warnings.append(f"seam_{seam:.3f}: redo failed: {call.error}")
            continue
        if call.parse_warning:
            redo_warnings.append(f"seam_{seam:.3f}: {call.parse_warning}")

        replacement = _offset_segments(
            call.segments,
            offset_sec=win_start,
            chunk_idx=200000 + successful_redos,
        )
        updated = [
            seg
            for seg in updated
            if float(seg["End"]) <= win_start or float(seg["Start"]) >= win_end
        ]
        updated.extend(replacement)
        updated.sort(key=lambda seg: (float(seg["Start"]), float(seg["End"])))
        updated, _ = _dedupe_overlap_segments(updated, overlap_seconds=scenario.overlap_seconds)
        successful_redos += 1

    stats = {
        "redo_request_count": successful_redos,
        "redo_latencies_sec": [round(v, 6) for v in redo_latencies],
        "redo_retry_count": redo_retries,
        "redo_warnings": redo_warnings,
    }
    return updated, stats


def _run_chunked_transcription(
    item: ManifestItem,
    scenario: ScenarioConfig,
    args: argparse.Namespace,
    variant_features: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], bool, str | None]:
    if scenario.chunk_minutes is None:
        raise ValueError("Chunked transcription requires chunk_minutes.")

    total_duration = _get_duration_seconds_ffprobe(item.path)
    chunk_seconds = float(scenario.chunk_minutes * 60)
    starts = _build_chunk_starts(total_duration, chunk_seconds, scenario.overlap_seconds)
    if "silence_aligned_boundaries" in variant_features:
        silence_points = _detect_silence_points_ffmpeg(
            item.path,
            noise_db=args.silence_noise_db,
            min_duration=args.silence_min_duration,
        )
        starts = _align_starts_to_silence(
            starts=starts,
            silence_points=silence_points,
            search_window_sec=args.silence_search_window_seconds,
            total_duration=total_duration,
        )
    chunks = _chunks_from_starts(starts, total_duration, chunk_seconds)

    primary_segments, primary_details = _run_chunks_once(
        item=item,
        scenario=scenario,
        args=args,
        chunks=chunks,
        variant_features=variant_features,
        chunk_index_offset=0,
    )
    parse_warnings: list[str] = list(primary_details["parse_warnings"])
    chunk_failures = int(primary_details["chunk_failures"])
    retries = int(primary_details["retry_count"])
    chunk_latencies = list(primary_details["chunk_latencies_sec"])
    seam_boundaries = list(primary_details["seam_boundaries_sec"])
    overlap_dropped = int(primary_details["overlap_dropped_segments"])
    chunk_count = int(primary_details["chunk_count"])
    consensus_matches = 0

    merged_segments = list(primary_segments)

    if "shifted_consensus" in variant_features:
        shift_seconds = chunk_seconds * float(args.shifted_offset_ratio)
        shifted_chunks = _build_shifted_chunks(
            total_duration=total_duration,
            chunk_seconds=chunk_seconds,
            overlap_seconds=scenario.overlap_seconds,
            shift_seconds=shift_seconds,
        )
        shifted_segments, shifted_details = _run_chunks_once(
            item=item,
            scenario=scenario,
            args=args,
            chunks=shifted_chunks,
            variant_features=variant_features - {"sentence_seam"},
            chunk_index_offset=1000,
        )
        # Shifted pass is advisory; failures should not invalidate the whole run.
        if shifted_segments and int(shifted_details["chunk_failures"]) == 0:
            merged_segments, consensus_matches = _consensus_merge_segments(
                primary_segments=merged_segments,
                shifted_segments=shifted_segments,
                overlap_seconds=scenario.overlap_seconds,
            )
            overlap_dropped += int(shifted_details["overlap_dropped_segments"])
            chunk_count += int(shifted_details["chunk_count"])
            retries += int(shifted_details["retry_count"])
            chunk_latencies.extend(shifted_details["chunk_latencies_sec"])
        else:
            parse_warnings.append("shifted_consensus: shifted pass failed, using primary only.")

    redo_stats = {
        "redo_request_count": 0,
        "redo_latencies_sec": [],
        "redo_retry_count": 0,
        "redo_warnings": [],
    }
    if "seam_micro_redo" in variant_features and merged_segments:
        merged_segments, redo_stats = _run_seam_micro_redo(
            item=item,
            scenario=scenario,
            args=args,
            segments=merged_segments,
            seam_boundaries=seam_boundaries,
            total_duration=total_duration,
        )
        retries += int(redo_stats["redo_retry_count"])
        chunk_latencies.extend(redo_stats["redo_latencies_sec"])
        parse_warnings.extend(redo_stats["redo_warnings"])

    merged_segments.sort(key=lambda seg: (float(seg["Start"]), float(seg["End"])))
    merged_segments, dropped_final = _dedupe_overlap_segments(
        merged_segments, overlap_seconds=scenario.overlap_seconds
    )
    overlap_dropped += dropped_final
    public_segments = _public_segments(merged_segments)

    p95_latency = None
    if chunk_latencies:
        sorted_lat = sorted(float(v) for v in chunk_latencies)
        idx = min(len(sorted_lat) - 1, math.ceil(0.95 * len(sorted_lat)) - 1)
        p95_latency = sorted_lat[idx]

    details = {
        "chunk_count": chunk_count,
        "chunk_failures": chunk_failures,
        "retry_count": retries,
        "chunk_latency_p95_sec": None if p95_latency is None else round(float(p95_latency), 6),
        "chunk_latencies_sec": [round(float(v), 6) for v in chunk_latencies],
        "total_duration_sec": round(total_duration, 6),
        "seam_boundaries_sec": [round(float(v), 3) for v in seam_boundaries],
        "overlap_dropped_segments": overlap_dropped,
        "parse_warnings": parse_warnings,
        "variant_features": sorted(variant_features),
        "consensus_matches": consensus_matches,
        "redo_request_count": int(redo_stats["redo_request_count"]),
    }
    success = bool(public_segments) and chunk_failures == 0
    error = None if success else "One or more chunk requests failed or no segments produced."
    return public_segments, details, success, error


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

    payload = {"segments": _public_segments(segments)}
    if raw_text is not None:
        payload["raw_text"] = raw_text
    if metadata is not None:
        payload["metadata"] = metadata

    _write_json(json_path, payload)
    _write_text(txt_path, _join_segment_content(_public_segments(segments)))


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
        by_scenario.setdefault(str(run["scenario"]), []).append(run)

    lines: list[str] = []
    lines.append("# Chunking Experiment Summary")
    lines.append("")
    lines.append(f"- Generated: {_utc_now_iso()}")
    lines.append(f"- Quality threshold (word drift): {threshold:.4f}")
    lines.append("")
    lines.append("| Scenario | Runs | Success | Mean Word Drift | Mean Seam Drift | Mean Char Drift | Mean RTF |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for scenario in sorted(by_scenario):
        rows = by_scenario[scenario]
        success_rows = [row for row in rows if row.get("success")]
        word_vals = [float(row["word_drift"]) for row in success_rows if row.get("word_drift") is not None]
        seam_vals = [float(row["seam_word_drift"]) for row in success_rows if row.get("seam_word_drift") is not None]
        char_vals = [float(row["char_drift"]) for row in success_rows if row.get("char_drift") is not None]
        rtf_vals = [float(row["rtf"]) for row in success_rows if row.get("rtf") is not None]
        lines.append(
            "| {scenario} | {runs} | {ok} | {word} | {seam} | {char} | {rtf} |".format(
                scenario=scenario,
                runs=len(rows),
                ok=len(success_rows),
                word=f"{statistics.mean(word_vals):.4f}" if word_vals else "NA",
                seam=f"{statistics.mean(seam_vals):.4f}" if seam_vals else "NA",
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
        and (float(run["word_drift"]) > threshold or float(run["word_drift"]) > 0.01)
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
        lines.append(f"- Word drift: {float(run['word_drift']):.4f}")
        if run.get("seam_word_drift") is not None:
            lines.append(f"- Seam word drift: {float(run['seam_word_drift']):.4f}")
        lines.append(f"- Char drift: {float(run['char_drift']):.4f}")
        if snippets:
            lines.extend(snippets)
        else:
            lines.append("- No diff snippets could be extracted.")
        lines.append("")
    _write_text(manual_review_path, "\n".join(lines) + "\n")


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
    variant_features: set[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    segments, details, success, error = _run_chunked_transcription(
        item=item,
        scenario=scenario,
        args=args,
        variant_features=variant_features,
    )
    elapsed = time.perf_counter() - started
    duration = details.get("total_duration_sec") or _get_duration_seconds_ffprobe(item.path)
    return {
        "file_id": item.file_id,
        "file_path": item.path,
        "scenario": scenario.name,
        "variant": args.variant,
        "success": success,
        "error": error,
        "duration_sec": round(float(duration), 6),
        "wall_time_sec": round(elapsed, 6),
        "rtf": round(elapsed / float(duration), 6) if duration else None,
        "chunks_processed_per_min": round(
            (float(details["chunk_count"]) / elapsed) * 60.0, 6
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
    variant_features: set[str],
) -> None:
    for workers in worker_counts:
        wave_start = time.perf_counter()
        wave_results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _run_single_scenario_for_item,
                    item=item,
                    scenario=scenario,
                    args=args,
                    variant_features=variant_features,
                )
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
            "variant": args.variant,
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
            "seam_word_drift": None,
            "seam_boundary_mae_sec": None,
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
                "variant": args.variant,
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
                "seam_word_drift": None,
                "seam_boundary_mae_sec": None,
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
        description="Run chunking quality and throughput experiments for VibeVoice ASR variants."
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
        "--variant",
        default="baseline_overlap_context",
        help=(
            "Variant alias or comma-separated feature list. "
            "Aliases: baseline_overlap_context, idea1_sentence_seam, "
            "idea2_seam_micro_redo, idea3_shifted_consensus, "
            "idea4_silence_aligned_boundaries, idea5_dynamic_lexicon."
        ),
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
        "--seam-window-seconds",
        type=float,
        default=60.0,
        help="Window around seam boundaries for sentence seam selection and seam metrics.",
    )
    parser.add_argument(
        "--micro-redo-window-seconds",
        type=float,
        default=90.0,
        help="Half-window size used for seam micro-redo requests.",
    )
    parser.add_argument(
        "--shifted-offset-ratio",
        type=float,
        default=0.5,
        help="Shift ratio for shifted-grid consensus relative to chunk size.",
    )
    parser.add_argument(
        "--silence-search-window-seconds",
        type=float,
        default=45.0,
        help="Search window around nominal boundaries for silence alignment.",
    )
    parser.add_argument(
        "--silence-noise-db",
        type=float,
        default=-35.0,
        help="silencedetect noise threshold in dB.",
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
        help="Max dynamic lexicon terms carried as hotwords.",
    )
    parser.add_argument(
        "--dynamic-lexicon-min-len",
        type=int,
        default=4,
        help="Minimum token length for dynamic lexicon candidates.",
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
        "--gold-artifact-dir",
        default=None,
        help="Reuse gold_full transcripts from an existing artifact directory.",
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Skip live full-length gold requests and require --gold-artifact-dir.",
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
    variant_features = _resolve_variant_features(args.variant)
    gold_artifact_dir = Path(args.gold_artifact_dir).resolve() if args.gold_artifact_dir else None
    if args.skip_gold and not gold_artifact_dir:
        raise ValueError("--skip-gold requires --gold-artifact-dir.")

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
            "variant": args.variant,
            "variant_features": sorted(variant_features),
            "gold_artifact_dir": str(gold_artifact_dir) if gold_artifact_dir else None,
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
        gold_source = "live"
        gold_result: ApiCallResult
        duration: float
        gold_started = time.perf_counter()
        if gold_artifact_dir:
            try:
                gold_result, duration = _load_gold_from_artifact(item, gold_artifact_dir)
                gold_source = "cache"
            except Exception as exc:  # noqa: BLE001
                if args.skip_gold:
                    print(f"[WARN] Cached gold missing for {item.file_id}: {exc}")
                    continue
                print(f"[WARN] Cached gold unavailable for {item.file_id}, falling back to live: {exc}")
                gold_result, duration = _run_full_transcription(item, args)
                gold_source = "live_fallback"
        else:
            print(f"[INFO] Running gold transcription for {item.file_id} :: {item.path}")
            gold_result, duration = _run_full_transcription(item, args)
        gold_elapsed = time.perf_counter() - gold_started

        gold_row = {
            "file_id": item.file_id,
            "file_path": item.path,
            "scenario": "gold_full",
            "variant": args.variant,
            "gold_source": gold_source,
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
            "seam_word_drift": 0.0 if gold_result.success else None,
            "seam_boundary_mae_sec": 0.0 if gold_result.success else None,
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
                "gold_source": gold_source,
            },
        )

        if not gold_result.success:
            print(f"[WARN] Gold run failed for {item.file_id}: {gold_result.error}")
            continue

        for scenario in quality_scenarios:
            print(f"[INFO] Running {scenario.name} for {item.file_id} (variant={args.variant})")
            run_started = time.perf_counter()
            segments, details, success, error = _run_chunked_transcription(
                item=item,
                scenario=scenario,
                args=args,
                variant_features=variant_features,
            )
            elapsed = time.perf_counter() - run_started
            comparison = _compare_to_gold(gold_result.segments, segments) if success else {}
            seam_metrics = (
                _compute_seam_metrics(
                    gold_segments=gold_result.segments,
                    cand_segments=segments,
                    seam_boundaries=details.get("seam_boundaries_sec", []),
                    seam_window_sec=args.seam_window_seconds,
                )
                if success
                else {"seam_word_drift": None, "seam_boundary_mae_sec": None}
            )
            row = {
                "file_id": item.file_id,
                "file_path": item.path,
                "scenario": scenario.name,
                "variant": args.variant,
                "success": success,
                "error": error,
                "duration_sec": round(float(details["total_duration_sec"]), 6),
                "wall_time_sec": round(elapsed, 6),
                "rtf": round(elapsed / float(details["total_duration_sec"]), 6)
                if details["total_duration_sec"]
                else None,
                "chunks_processed_per_min": round(
                    (float(details["chunk_count"]) / elapsed) * 60.0, 6
                )
                if elapsed > 0
                else None,
                "chunk_count": details["chunk_count"],
                "chunk_failures": details["chunk_failures"],
                "retry_count": details["retry_count"],
                "chunk_latency_p95_sec": details["chunk_latency_p95_sec"],
                "word_drift": comparison.get("word_drift"),
                "char_drift": comparison.get("char_drift"),
                "seam_word_drift": seam_metrics.get("seam_word_drift"),
                "seam_boundary_mae_sec": seam_metrics.get("seam_boundary_mae_sec"),
                "segment_count_delta": comparison.get("segment_count_delta"),
                "boundary_mae_sec": comparison.get("boundary_mae_sec"),
                "quality_pass": (
                    comparison.get("word_drift") is not None
                    and float(comparison["word_drift"]) <= args.quality_threshold
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
        print(
            "[INFO] Running throughput matrix using scenario: "
            f"{selected_throughput_scenario} (variant={args.variant})"
        )
        _run_throughput_matrix(
            items=items,
            scenario=scenario_obj,
            worker_counts=args.throughput_workers,
            args=args,
            output_dir=output_dir,
            runs_jsonl_path=runs_jsonl_path,
            collected_runs=all_runs,
            variant_features=variant_features,
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

    print("[INFO] Experiment complete.")
    print(f"[INFO] Artifacts directory: {output_dir}")
    print("[INFO] Summary:")
    print(summary_text)


if __name__ == "__main__":
    main()
