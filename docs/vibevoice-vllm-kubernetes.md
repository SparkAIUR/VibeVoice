# VibeVoice vLLM ASR on Kubernetes

This guide mirrors the Docker deployment and mounts host files into the pod:
- VibeVoice source: `/shared/projects/VibeVoice` -> `/app`
- Audio corpus: `/data/audio-calls` -> `/data/audio-calls` (read-only)

Use the manifest at:

`deploy/kubernetes/vibevoice-vllm-hostpath.yaml`

## 1) Prerequisites

- NVIDIA GPU node with Kubernetes NVIDIA device plugin installed.
- Node contains:
  - `/shared/projects/VibeVoice`
  - `/data/audio-calls`
- `kubectl` access to the target cluster.

## 2) Customize the manifest

Edit these fields before applying:

- `spec.template.spec.nodeSelector.kubernetes.io/hostname`
  - Default is `hpldgx02`.
- `volumes[].hostPath.path`
  - Confirm host paths exist on your GPU node.
- `resources.{requests,limits}`
  - Tune CPU/memory/GPU for your environment.
- `args` command flags
  - Example currently uses:
    - `--max-model-len 65536`
    - `--max-num-seqs 8`
    - `--gpu-memory-utilization 0.90`
    - `--enforce-eager`

## 3) Deploy

```bash
kubectl apply -k deploy/kubernetes
```

Or:

```bash
kubectl apply -f deploy/kubernetes/vibevoice-vllm-hostpath.yaml
```

## 4) Verify readiness

```bash
kubectl -n vibevoice get pods -w
kubectl -n vibevoice logs -f deploy/vibevoice-vllm
```

Check API from local machine:

```bash
kubectl -n vibevoice port-forward svc/vibevoice-vllm 8000:8000
curl -sf http://127.0.0.1:8000/v1/models
```

## 5) Run transcription tests from repo root

```bash
python3 vllm_plugin/tests/test_api.py /data/audio-calls/tethr-calls/db3nc18hf-es.wav --api_url http://127.0.0.1:8000
```

## 6) Update rollout

After editing manifest values:

```bash
kubectl apply -k deploy/kubernetes
kubectl -n vibevoice rollout status deploy/vibevoice-vllm
```

## 7) Cleanup

```bash
kubectl delete -k deploy/kubernetes
```

## Notes

- The deployment uses `hostIPC: true` to match Docker `--ipc=host`.
- `/dev/shm` is also mounted as memory-backed `emptyDir`.
- If your cluster blocks `hostIPC`, remove it and increase `shm` size as needed.
