# Long Audio Chunking Experiment

This runbook evaluates chunking quality drift against full-length transcription.

## 1) Clean and relaunch container on `hpldgx02`

```bash
ssh hpldgx02 '
  docker rm -f vibevoice-vllm >/dev/null 2>&1 || true
  cd /shared/projects/VibeVoice
  docker run -d --gpus all --name vibevoice-vllm \
    --ipc=host \
    -p 8000:8000 \
    -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=20 \
    -e VLLM_MEDIA_LOADING_THREAD_COUNT=8 \
    -e VIBEVOICE_MAX_AUDIO_DURATION=7200 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -v /shared/projects/VibeVoice:/app \
    -v /data/audio-calls:/data/audio-calls:ro \
    -w /app \
    --entrypoint bash \
    vllm/vllm-openai:v0.14.1 \
    -c "python3 /app/vllm_plugin/scripts/start_server.py --max-model-len 65536 --max-num-seqs 8 --gpu-memory-utilization 0.90 --enforce-eager"
'
```

Check readiness:

```bash
ssh hpldgx02 'until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 2; done; echo ready'
```

## 2) Run pilot quality + throughput matrix

```bash
ssh hpldgx02 '
  cd /shared/projects/VibeVoice
  python3 vllm_plugin/experiments/run_chunking_experiment.py \
    --manifest vllm_plugin/experiments/manifests/pilot_long_audio_6.json \
    --output-dir artifacts/chunking_experiment \
    --api-url http://127.0.0.1:8000 \
    --chunk-minutes 10 20 30 \
    --overlap-seconds 30 \
    --context-tail-chars 800 \
    --quality-threshold 0.02 \
    --run-throughput \
    --throughput-workers 1 2 4 6
'
```

Artifacts are written under:

`artifacts/chunking_experiment/<timestamp>/`

Key files:
- `manifest_used.json`
- `runs.jsonl`
- `summary.csv`
- `summary.md`
- `manual_review.md`
- `expert_review/rubric.md`
- `expert_review/review_template.json`
- `expert_review/review_template.md`
- `transcripts/<file_id>/<scenario>.json|txt`

To turn a completed expert review into a reranked scoreboard:

```bash
python3 vllm_plugin/experiments/summarize_expert_review.py \
  --review-json artifacts/chunking_experiment/<timestamp>/expert_review/review_template.json
```

If a `gold_full` transcript is clearly invalid or truncated, set `review.exclude_from_ranking=true`
and `review.exclusion_reason` in `review_template.json` before summarizing. Excluded items stay in
the audit trail but are omitted from the ranked averages.

## 3) Parallel A/B against Spark + production endpoint

This runs identical experiments in parallel against:
- Spark/local endpoint: `http://127.0.0.1:8000`
- Production endpoint: `https://vibevoice-asr.internal.nu-dev.co`

Manifest:
- `vllm_plugin/experiments/manifests/ab_parallel_spanish_2.json`

### 3.1 Launch paired quality runs (concurrent)

```bash
ssh hpldgx02 '
  cd /shared/projects/VibeVoice

  nohup python3 -u vllm_plugin/experiments/run_chunking_experiment.py \
    --manifest vllm_plugin/experiments/manifests/ab_parallel_spanish_2.json \
    --output-dir artifacts/chunking_experiment/ab_spark \
    --api-url http://127.0.0.1:8000 \
    --chunk-minutes 10 20 \
    --scenario-modes no_overlap overlap_context \
    --quality-threshold 0.02 \
    --max-tokens-gold 6144 \
    --max-tokens-chunk 1024 \
    --timeout-seconds 1800 \
    --max-retries 1 \
    > artifacts/chunking_experiment/ab_spark_quality.log 2>&1 < /dev/null &

  nohup python3 -u vllm_plugin/experiments/run_chunking_experiment.py \
    --manifest vllm_plugin/experiments/manifests/ab_parallel_spanish_2.json \
    --output-dir artifacts/chunking_experiment/ab_prod \
    --api-url https://vibevoice-asr.internal.nu-dev.co \
    --chunk-minutes 10 20 \
    --scenario-modes no_overlap overlap_context \
    --quality-threshold 0.02 \
    --max-tokens-gold 6144 \
    --max-tokens-chunk 1024 \
    --timeout-seconds 1800 \
    --max-retries 1 \
    > artifacts/chunking_experiment/ab_prod_quality.log 2>&1 < /dev/null &
'
```

### 3.2 Launch paired throughput runs (concurrent, worker sweep up to 8)

Use the same scenario on both sides (example uses `chunk_10m_no_overlap`):

```bash
ssh hpldgx02 '
  cd /shared/projects/VibeVoice

  nohup python3 -u vllm_plugin/experiments/run_chunking_experiment.py \
    --manifest vllm_plugin/experiments/manifests/ab_parallel_spanish_2.json \
    --output-dir artifacts/chunking_experiment/ab_spark_tp \
    --api-url http://127.0.0.1:8000 \
    --chunk-minutes 10 20 \
    --scenario-modes no_overlap overlap_context \
    --quality-threshold 0.02 \
    --max-tokens-gold 6144 \
    --max-tokens-chunk 1024 \
    --timeout-seconds 1800 \
    --max-retries 1 \
    --run-throughput \
    --throughput-scenario chunk_10m_no_overlap \
    --throughput-workers 1 2 4 8 \
    > artifacts/chunking_experiment/ab_spark_throughput.log 2>&1 < /dev/null &

  nohup python3 -u vllm_plugin/experiments/run_chunking_experiment.py \
    --manifest vllm_plugin/experiments/manifests/ab_parallel_spanish_2.json \
    --output-dir artifacts/chunking_experiment/ab_prod_tp \
    --api-url https://vibevoice-asr.internal.nu-dev.co \
    --chunk-minutes 10 20 \
    --scenario-modes no_overlap overlap_context \
    --quality-threshold 0.02 \
    --max-tokens-gold 6144 \
    --max-tokens-chunk 1024 \
    --timeout-seconds 1800 \
    --max-retries 1 \
    --run-throughput \
    --throughput-scenario chunk_10m_no_overlap \
    --throughput-workers 1 2 4 8 \
    > artifacts/chunking_experiment/ab_prod_throughput.log 2>&1 < /dev/null &
'
```

### 3.3 Compare both endpoints against production gold

Choose artifact dirs from each run and compare:

```bash
ssh hpldgx02 '
  cd /shared/projects/VibeVoice
  python3 vllm_plugin/experiments/compare_ab_endpoints.py \
    --prod-dir artifacts/chunking_experiment/ab_prod/<prod_timestamp> \
    --spark-dir artifacts/chunking_experiment/ab_spark/<spark_timestamp> \
    --output-dir artifacts/chunking_experiment/ab_compare \
    --quality-threshold 0.02
'
```

Comparator outputs:
- `ab_quality_summary.csv`
- `ab_quality_summary.md`
- `ab_manual_review.md`
- `expert_review/rubric.md`
- `expert_review/review_template.json`
- `expert_review/review_template.md`
- `ab_throughput_summary.csv` (when throughput rows are present)
- `ab_throughput_summary.md` (when throughput rows are present)
- `ab_recommendation.json`

## 4) Systematic idea matrix (5 ideas + greedy combos)

This runs the two-phase matrix on the production endpoint:
- Phase 1: baseline + 5 single ideas + greedy combinations on 8-file stratified set
- Phase 2: baseline + finalists on full 23-file set

Default manifests:
- `vllm_plugin/experiments/manifests/endpoint_02172026_phase1_8.json`
- `vllm_plugin/experiments/manifests/endpoint_02172026_phase2_23.json`

```bash
cd /shared/projects/VibeVoice
python3 vllm_plugin/experiments/run_idea_matrix.py \
  --api-url https://vibevoice-asr.internal.nu-dev.co \
  --phase1-manifest vllm_plugin/experiments/manifests/endpoint_02172026_phase1_8.json \
  --phase2-manifest vllm_plugin/experiments/manifests/endpoint_02172026_phase2_23.json \
  --output-dir artifacts/chunking_experiment/idea_matrix \
  --chunk-minutes 30 \
  --quality-threshold 0.02 \
  --perf-budget-drop 0.20 \
  --max-workers 16 \
  --max-tokens 32768 \
  --max-tokens-gold 32768 \
  --max-tokens-chunk 24576 \
  --timeout-seconds 7200 \
  --max-retries 1
```

Key outputs under `artifacts/chunking_experiment/idea_matrix/<timestamp>/`:
- `phase1/variant_scoreboard.csv|md`
- `phase1/combo_scoreboard.csv|md`
- `phase1/expert_review/rubric.md`
- `phase1/expert_review/review_template.json`
- `phase2/variant_scoreboard.csv|md`
- `phase2/expert_review/rubric.md`
- `phase2/expert_review/review_template.json`
- `final_recommendation.json`

Variant runner used by the matrix:
- `vllm_plugin/experiments/run_chunking_variant_experiment.py`

Rubric reference:
- `docs/transcript_evaluation_rubric.md`

## 5) Long-tail finalist validation

Use this after narrowing down the finalists. It runs only the carried-forward variants on the longest calls, reuses baseline gold transcripts, and generates a root-level expert-review bundle for subjective reranking.

Manifest:
- `vllm_plugin/experiments/manifests/endpoint_02172026_long_tail_6.json`

Runner:
- `vllm_plugin/experiments/run_finalist_validation.py`
- `vllm_plugin/experiments/merge_finalist_validation.py` for interrupted-run recovery

Example:

```bash
python3 vllm_plugin/experiments/run_finalist_validation.py \
  --api-url http://vibevoice-asr.nu-dev.io:8000 \
  --manifest vllm_plugin/experiments/manifests/endpoint_02172026_long_tail_6.json \
  --output-dir artifacts/chunking_experiment/endpoint_finalists_long_tail \
  --baseline-gold-artifact-dir artifacts/chunking_experiment/endpoint_finalists_long_tail_test/20260409T221252Z/runs/baseline_overlap_context/20260409T221252Z \
  --baseline-skip-gold \
  --chunk-minutes 30 \
  --max-tokens 32768 \
  --max-tokens-gold 32768 \
  --max-tokens-chunk 24576 \
  --timeout-seconds 7200 \
  --max-retries 1
```

`--baseline-gold-artifact-dir` + `--baseline-skip-gold` is the preferred path for rapid iteration cycles because it avoids re-running expensive full-length gold requests when trustworthy cached gold already exists.

Outputs under `artifacts/chunking_experiment/endpoint_finalists_long_tail/<timestamp>/`:
- `variant_scoreboard.csv|md`
- `expert_review/rubric.md`
- `expert_review/review_template.json`
- `expert_review/review_template.md`
- `runs/<variant>/<timestamp>/...`

For long-tail reviews, exclude any item whose `gold_full` transcript is not trustworthy rather than
forcing a subjective score against a bad reference.

If a finalist run is interrupted and you need to merge a resumed shard back into one clean result set:

```bash
python3 vllm_plugin/experiments/merge_finalist_validation.py \
  --run-root artifacts/chunking_experiment/endpoint_finalists_long_tail/<timestamp> \
  --variant-artifact baseline_overlap_context=/abs/path/to/baseline_artifact \
  --variant-artifact idea3_shifted_consensus=/abs/path/to/idea3_partial_artifact \
  --variant-artifact idea3_shifted_consensus=/abs/path/to/idea3_resume_artifact \
  --variant-artifact idea5_dynamic_lexicon=/abs/path/to/idea5_artifact
```

This writes:
- root `variant_scoreboard.csv|md`
- root `expert_review/` bundle
- merged per-variant artifacts under `merged/`
- `merge_report.json`

### 5.2 Refresh Long-Call Gold With Chunked Reference

For the longest calls, a single live `gold_full` request can still be untrustworthy even if the
endpoint accepts the audio, because audio tokens consume most of the 64K context window and leave
too little completion budget for a full transcript. In that case:

1. Generate a chunked reference transcript for the affected files, typically with a stronger
   finalist such as `idea5_dynamic_lexicon` on `chunk_10m_overlap_context`.
2. Merge the per-file reference artifacts into one reference root.
3. Refresh the long-tail finalist run so `gold_full` points at the chunked reference and candidate
   metrics are recomputed against it.

Helper:
- `vllm_plugin/experiments/refresh_reference_gold.py`

Example:

```bash
python3 vllm_plugin/experiments/refresh_reference_gold.py \
  --run-root artifacts/chunking_experiment/endpoint_finalists_long_tail_test/<timestamp> \
  --reference-artifact-dir artifacts/chunking_experiment/endpoint_reference_gold_idea5_10m_merged/<timestamp> \
  --reference-scenario chunk_10m_overlap_context \
  --file-ids en_78m_db4cc2fnm en_82m_eb4dbbbhj en_82m_db4dc2myy en_84m_db4dc2m8r en_114m_db4cc118k \
  --scenario chunk_30m_overlap_context \
  --chunk-minutes 30 \
  --overlap-seconds 30 \
  --seam-window-seconds 60
```

After refresh:

```bash
python3 vllm_plugin/experiments/summarize_expert_review.py \
  --review-json artifacts/chunking_experiment/endpoint_finalists_long_tail_test/<timestamp>/expert_review/review_template.json
```

## 6) Candidate next methods

If long-tail validation confirms the current finalists are still the best, the next ideas worth testing are:

- Boundary rescoring with sentence completeness:
  choose the overlap handoff point that maximizes semantic continuity and avoids cutting named entities or number phrases.
- Selective second-pass repair:
  run a lightweight detector for suspicious windows (numbers, entity drift, abrupt repetition, malformed seam text) and only re-transcribe those windows.
- Call-global entity memory:
  build a rolling lexicon of providers, names, phone numbers, and repeated terms from earlier chunks, then use it to bias later chunks and seam selection.
- Dual-grain chunking:
  transcribe with large chunks for stability, then run smaller overlap windows only near seams and use local alignment to keep the better phrasing.
- Diarization / turn-aware seams:
  align chunk boundaries to speaker turns or pauses so stitched transcripts are less likely to split semantic units mid-utterance.

## 7) Continuous Iteration Loop

Use the iteration program manifest to keep the next cycle, validation ladder, and durable notes in sync.

Program manifest:
- `vllm_plugin/experiments/manifests/asr_iteration_program.json`

Coordinator:
- `vllm_plugin/experiments/run_iteration_cycle.py`

Durable notes:
- `refs/notes/asr_iteration_program.md`
- `refs/notes/asr_hypothesis_backlog.md`
- `refs/notes/asr_iteration_<cycle_id>.md`

Hook gate (required before execution):
- Every new backlog idea must have a concrete `runnable_variant` in the program manifest.
- Each runnable variant must be backed by a feature hook in `vllm_plugin/experiments/run_chunking_variant_experiment.py`.
- If `runnable_variant` is missing, `run_iteration_cycle.py execute` will stop the cycle instead of running partial experiments.

### 7.1 Prepare the next cycle

This ranks the backlog, selects up to 3 ideas, writes the cycle spec, and refreshes the program notes.

```bash
python3 vllm_plugin/experiments/run_iteration_cycle.py prepare \
  --program-manifest vllm_plugin/experiments/manifests/asr_iteration_program.json \
  --cycle-id cycle_20260410_cycle01
```

Outputs:
- `artifacts/chunking_experiment/iteration_cycles/<cycle_id>/cycle_spec.json`
- refreshed `refs/notes/asr_iteration_program.md`
- refreshed `refs/notes/asr_hypothesis_backlog.md`
- `refs/notes/asr_iteration_<cycle_id>.md`

### 7.2 Execute the validation ladder

This runs the selected champion + challengers through:
- smoke
- sentinel
- long-tail
- throughput

```bash
python3 vllm_plugin/experiments/run_iteration_cycle.py execute \
  --cycle-spec artifacts/chunking_experiment/iteration_cycles/<cycle_id>/cycle_spec.json
```

Important:
- `execute` only runs challengers that are already wired into the harness with `runnable_variant`.
- If a selected idea is still design-only, the cycle remains documented but blocked until the implementation adds a runnable variant hook.

For unattended operation, use the supervisor (polls every 15 minutes, relaunches execute if needed, summarizes stage reviews when scores exist, and finalizes the cycle):

```bash
python3 vllm_plugin/experiments/run_iteration_supervisor.py \
  --cycle-spec artifacts/chunking_experiment/iteration_cycles/<cycle_id>/cycle_spec.json \
  --poll-seconds 900 \
  --max-hours 8
```

Supervisor artifacts:
- `artifacts/chunking_experiment/iteration_cycles/<cycle_id>/logs/supervisor.log`
- `artifacts/chunking_experiment/iteration_cycles/<cycle_id>/supervisor_state.json`

### 7.3 Finalize after expert review

After filling each stage's `expert_review/review_template.json`, summarize it and finalize the cycle:

```bash
python3 vllm_plugin/experiments/summarize_expert_review.py \
  --review-json artifacts/chunking_experiment/iteration_cycles/<cycle_id>/stages/long_tail/<run_timestamp>/expert_review/review_template.json

python3 vllm_plugin/experiments/run_iteration_cycle.py finalize \
  --cycle-spec artifacts/chunking_experiment/iteration_cycles/<cycle_id>/cycle_spec.json
```

`finalize` will:
- evaluate smoke, sentinel, long-tail, and throughput gates
- decide whether a challenger should replace the champion
- update the backlog statuses
- refresh the durable notes
- increment the production-readiness counter only when the champion clears the stop gate

### 7.4 Stop condition

Do not stop the loop until the same champion clears the production gate in two consecutive finalized cycles:
- mean expert score `>= 85` on the long-tail set
- zero critical issues
- zero meaning-loss rows
- zero `not_acceptable` long-tail rows
- no unresolved completeness-loss failures on `>60m` calls
- throughput/concurrency validated at workers `2` and `4`

When that happens, freeze the winner and write the final report in:
- `refs/notes/asr_production_readiness_report.md`
