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
- `transcripts/<file_id>/<scenario>.json|txt`
