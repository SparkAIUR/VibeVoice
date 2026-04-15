# Transcript Evaluation Rubric

This rubric standardizes expert review of ASR transcripts so method selection is not driven by word drift alone.

## Goal

Use expert review as the primary signal for deciding whether a chunking method is good enough to replace the current baseline. Metrics such as word drift, seam drift, and RTF remain useful, but they are supporting signals.

## Scoring Model

Score each transcript against the `gold_full` transcript on a 0-5 integer scale for each criterion:

- `semantic_fidelity` (35%): does the candidate preserve the same meaning, decisions, and intent?
- `critical_details` (25%): are names, providers, numbers, dates, consent phrases, and verification/disposition details preserved?
- `completeness` (15%): are there material omissions or additions?
- `boundary_coherence` (10%): are chunk seams smooth, without duplication or clipping?
- `readability` (10%): is the transcript easy for an operator or QA reviewer to read?
- `noise_handling` (5%): are silence/noise/unintelligible markers used appropriately?

The weighted score is normalized to a 0-100 scale.

## Severity

Record issue counts separately:

- `critical`: meaning-changing or business-impacting error. Any critical issue makes the transcript `not_acceptable`.
- `major`: important trust-reducing error that does not fully invert meaning.
- `minor`: cosmetic or low-impact issue.

Also record whether there is explicit `meaning_loss`.

If the `gold_full` transcript is clearly invalid, truncated, or otherwise unusable as a reference, do not force a numeric score. Set:

- `exclude_from_ranking: true`
- `exclusion_reason: <why the gold is unusable>`

## Verdict Bands

- `equivalent`: score >= 90, no critical issues, no meaning loss
- `acceptable`: score >= 80, no critical issues, no meaning loss
- `borderline`: score >= 70, no critical issues, but visible quality concerns remain
- `not_acceptable`: score < 70, or any critical issue, or meaning loss

## Review Workflow

1. Open the generated review bundle under `expert_review/`.
2. Read the rubric in `expert_review/rubric.md`.
3. Fill scores and notes in `expert_review/review_template.json`.
   If a gold transcript is unusable, set the exclusion fields instead of scoring that item.
4. Run:

```bash
python3 vllm_plugin/experiments/summarize_expert_review.py \
  --review-json <artifact_dir>/expert_review/review_template.json
```

5. Use `expert_review/scoreboard.csv` and `expert_review/scoreboard.md` as the primary ranking signal.

## Tracking Over Time

For every future experiment, keep:

- the per-item expert review JSON
- the expert review scoreboard
- the metric scoreboard

This gives a stable record of:

- whether methods preserve business meaning
- whether improvements are real or only metric-visible
- whether throughput gains come with unacceptable semantic regressions
- which items were excluded because the reference transcript was not trustworthy
