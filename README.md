# Speech MLOps: Whisper ASR Backend with Time-Series Monitoring

A production-grade Automatic Speech Recognition (ASR) backend that transcribes audio asynchronously using OpenAI's Whisper (via faster-whisper / CTranslate2), with full MLOps tooling: Prometheus monitoring, drift detection, MLflow experiment tracking, load testing, and a statistical evaluation framework with paired bootstrap model comparison.

## Architecture

```
Client → FastAPI (upload + enqueue) → Redis (Celery broker) → Worker → PostgreSQL (source of truth)
                                                            ↓
                                              ffmpeg → VAD → chunking → Whisper
                                                            ↓
                                              Prometheus ← /metrics (from DB)
                                                ↓
                                           Grafana dashboards
                                                ↓
                                           Drift monitor (KS test on confidence signals)
```

**Tech stack:** FastAPI + Celery + Redis + PostgreSQL + faster-whisper (CTranslate2) + Silero VAD + Prometheus/Grafana + MLflow + Locust. S3/MinIO optional for storage. Docker Compose orchestrates everything.

---

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt
cp .env.example .env            # edit DATA_ROOT, set MODEL_BACKEND=whisper for real ASR
docker compose up -d            # PostgreSQL + Redis
alembic upgrade head
python -m scripts.create_user alice   # save the printed API key
uvicorn app.api.main:app --port 8000                          # terminal 2
celery -A app.worker.celery_app:celery_app worker -B -l info -c 1  # terminal 3
pytest -q                       # unit + API tests (no Docker needed)
```

---

## Project Phases

### Phase 1: Infrastructure (Storage, Database, Auth, API)

Built the foundation: a pluggable storage backend (local disk and S3/MinIO) with immutable, content-addressed objects; SQLAlchemy models with PostgreSQL; API key authentication (SHA-256 hashed); and the initial Alembic migration.

Key design decisions:
- **Immutable storage objects**: same key can't be overwritten; uploads deduplicated by per-user checksum
- **Path traversal protection**: storage keys validated against `..`, absolute paths, backslashes
- **Atomic writes**: local backend uses `os.replace` so readers never see half-written files

### Phase 2: Async Pipeline with Stub Model

Implemented the full async job pipeline using a **stub model** (sleeps, returns dummy text, fails on demand) so the infrastructure could be developed and load-tested without a GPU.

**Pipeline flow:** Upload → validate (ffprobe) → store → enqueue (Celery) → worker claims run → transcribe → record result in PostgreSQL.

**Reliability patterns:**
- **Compare-and-set idempotency**: `claim_run` atomically transitions queued → processing; duplicate deliveries are no-ops
- **Exponential backoff retries**: 2s, 4s, 8s... capped at 60s; permanent errors (corrupt audio) fail immediately
- **Reaper safety net**: runs every 60s, recovers from dead workers (processing too long → requeue) and lost broker messages (queued too long → re-enqueue)
- **Backpressure at two levels**: per-user concurrency limit (DB lock) + global queue depth (Redis LLEN)
- **`task_acks_late=True`**: ack after task finishes so a crashed worker's job is redelivered
- **`worker_prefetch_multiplier=1`**: fair scheduling for long-running jobs

**Load testing (Locust):** Tested with two user types — SubmitterUser (fire-and-forget, measures API acceptance rate) and FullCycleUser (upload → transcribe → wait, measures end-to-end latency). 429 backpressure responses counted separately, not as failures. All load test stages (A through E) confirmed zero data loss, with the queue draining completely after load stopped.

### Phase 4: Real Whisper Model

Replaced the stub with **faster-whisper** (CTranslate2 backend) running on GPU (CUDA, int8_float16). The pipeline now:

1. Fetches/caches a resampled 16 kHz mono WAV (content-addressed by audio_id + recipe)
2. Runs **Silero VAD** to find speech segments
3. **Chunks** speech segments (force-split long segments, pack neighbors up to 30s)
4. Transcribes each chunk independently with `condition_on_previous_text=False`
5. Computes audio quality signals: input RMS, silence ratio, per-chunk avg_logprob and no_speech_prob
6. Measures real-time factor (RTF = compute time / audio time) excluding model loading

The stub and real model both implement the same `ModelWrapper` ABC, so the rest of the system never knows which one it's talking to. The stub's `needs_audio=False` flag lets the pipeline skip decode/VAD entirely for cheap load testing.

### Phase 5: Evaluation Framework

#### LibriSpeech Baseline

Used LibriSpeech test-clean (200 utterances, 3,993 reference words, spread across many speakers) as the evaluation dataset. The manifest CSV carries `path` and `reference` columns.

#### Clean Evaluation Grid

Ran a 4 × 3 grid: model sizes (tiny, base, small, medium) × simulated sample rates (16, 8, 4 kHz) on 200 clean files. All 12 configurations completed with 0 failed runs.

| Model | 16 kHz | 8 kHz | 4 kHz | RTF (16kHz) |
|-------|--------|-------|-------|-------------|
| tiny  | 7.81%  | 8.59% | 23.32%| 0.061       |
| base  | 6.49%  | 7.01% | 14.68%| 0.060       |
| small | 4.21%  | 4.43% | 8.16% | 0.078       |
| medium| 3.66%  | 4.28% | 6.29% | 0.144       |

(WER with 95% bootstrap confidence intervals; RTF = total compute time / total audio time)

#### Paired Bootstrap Model Comparison

Used paired bootstrap resampling (5,000 iterations, same files for both models) to determine whether differences are statistically significant:

| Comparison | Δ WER (pp) | 95% CI | Verdict |
|-----------|-----------|--------|---------|
| tiny → base (16kHz) | -1.33 | [-2.31, -0.35] | base reliably better |
| base → small (16kHz) | -2.28 | [-3.01, -1.57] | small reliably better |
| small → medium (16kHz) | -0.55 | [-1.16, +0.05] | cannot tell apart |
| small 16kHz → 8kHz | +0.23 | [-0.16, +0.60] | cannot tell apart (telephone OK) |
| small 16kHz → 4kHz | +3.96 | [+3.14, +4.83] | 4kHz reliably worse |

Key findings:
- **Model size matters up to small**: tiny → base → small are reliably different. Small → medium is not distinguishable on 200 files.
- **8 kHz is fine**: downgrading to telephone bandwidth (8 kHz) doesn't significantly hurt WER. 4 kHz does.
- **Speed trade-off**: tiny/base run ~10x real-time; medium is ~7x real-time but not reliably better than small.

#### White Noise Sweep

Created noisy test sets at 20, 10, 5, and 0 dB SNR (white noise) and evaluated base, small, medium at 16 kHz:

| SNR | base | small | medium |
|-----|------|-------|--------|
| clean | 6.49% | 4.21% | 3.66% |
| 20 dB | 7.04% | 4.78% | 4.08% |
| 10 dB | 11.85% | 7.14% | 5.21% |
| 5 dB | 20.89% | 13.22% | 8.19% |
| 0 dB | 40.67% | 23.57% | 16.68% |

**WER rises sharply between 10 dB and 5 dB** for all models. At 0 dB (noise as loud as speech), even the medium model degrades to 16.7% WER. The medium model is consistently the most noise-robust.

#### Proxy Signal Analysis (Research Question)

The core research question: **do confidence/audio signals predict transcription errors?**

Pooled 4,800 per-file results across clean + noisy conditions and computed Spearman rank correlations between each signal and per-file WER:

| Signal | Clean-only ρ | Pooled ρ | Interpretation |
|--------|-------------|----------|----------------|
| avg_logprob | -0.527 | **-0.605** | Strong: lower confidence → more errors |
| no_speech_prob | 0.150 | 0.272 | Moderate: more no-speech → more errors |
| input_rms | -0.001 | 0.171 | Weak: RMS rises with noise, weakly tied to errors |
| silence_ratio | 0.101 | 0.103 | Weak |

At 0 dB SNR, avg_logprob correlation reaches **-0.800** — very strong. This confirms that decoder confidence (avg_logprob) is a useful proxy for transcription quality without needing reference text.

### Phase 6: Monitoring and Drift Detection

#### Prometheus Metrics

All metrics are derived from PostgreSQL at scrape time (not per-process), so they work regardless of how many API/worker processes exist. Exposes:
- Queue depth (Redis LLEN)
- Runs by status (queued/processing/done/failed)
- Window statistics: mean, p50, p95 for queue_wait_s, processing_s, RTF, avg_logprob, no_speech_prob, input_rms, silence_ratio

#### Alert Rules

Configured Prometheus alerts for: API down, high queue wait (p95 > 60s), high failure ratio (>10%), low confidence (avg_logprob < -1.0), and input drift detected.

#### Drift Monitor

A separate process (port 9102) that compares a **reference window** (runs tagged "baseline") against a **current window** (last N minutes) using two-sample Kolmogorov-Smirnov tests on 5 signals (avg_logprob, no_speech_prob, rtf, input_rms, silence_ratio). Drift = statistically significant (p < 0.01) AND large effect size (KS ≥ 0.3).

The drift demo uses a dedicated traffic user (to avoid the dedup trap where re-uploaded files keep their old dataset_tag). Send clean baseline traffic, wait, then send noisy traffic — the monitor detects the distribution shift in confidence signals.

---

## Directory Structure

```
app/
  config.py          # Single source of truth for all settings (Pydantic)
  pipeline.py         # ffmpeg → VAD → chunking → Whisper
  metrics.py          # Prometheus metrics from PostgreSQL
  api/                # FastAPI routes: audio upload, runs, health, metrics
  db/                 # SQLAlchemy models, repositories (all SQL), session
  services/           # Business logic: audio upload, run creation, error types
  models/             # ModelWrapper ABC, stub, faster-whisper implementation
  preprocessing/      # audio_io (ffmpeg/ffprobe), chunking, VAD (Silero)
  storage/            # StorageBackend ABC, local disk, S3/MinIO
  worker/             # Celery config, tasks (process_run, reap_stuck_runs)
eval/
  prepare_librispeech.py  # Download + build manifest
  make_noisy.py           # Create noisy copies at target SNR (white or real noise)
  run_eval.py             # Experiment grid → WER, RTF, confidence, MLflow
  metrics.py              # WER (jiwer), word_errors, bootstrap CIs, paired bootstrap
  compare_models.py       # Paired bootstrap: is model B reliably better than A?
  analyze_proxy.py        # Spearman ρ between confidence signals and WER
  plot_results.py         # Charts from MLflow: WER vs rate, WER vs latency, WER vs SNR
  send_traffic.py         # Steady traffic stream for monitoring/drift demo
  client.py               # HTTP client for the ASR backend
  transcribe_one.py       # Quick single-file transcription + WER
monitor/
  drift.py              # KS-test drift detector (Prometheus exporter on :9102)
scripts/
  create_user.py        # Create API users (SHA-256 hashed keys)
  plot_backlog.py       # Queue behavior under load (arrival vs completion rate)
  verify_drain.py       # Post-load verification: everything reached terminal state?
loadtests/
  locustfile.py         # Locust load test (SubmitterUser + FullCycleUser)
tests/                  # 44 tests: chunking, pipeline, drift/WER, API lifecycle, worker logic, storage
prometheus/             # Prometheus config + alert rules
grafana/                # Grafana datasource provisioning
```

---

## Key Design Patterns

1. **DB as source of truth** — PostgreSQL holds all state; Celery is just a message broker (no result backend)
2. **Compare-and-set idempotency** — `claim_run` only transitions queued→processing atomically
3. **Immutable storage** — objects can't be overwritten; dedup by checksum
4. **Lazy heavy imports** — ML libraries only imported in workers, not the API process
5. **Backpressure at two levels** — per-user concurrency (DB lock) + global queue depth (Redis)
6. **Reaper safety net** — recovers from dead workers and lost broker messages every 60s
7. **Content-addressed caching** — derived WAVs cached by audio_id + recipe
8. **Config validation** — unknown config keys rejected; allowed keys whitelisted
9. **Corpus RTF** — total compute / total audio (honest throughput; mean of per-file RTFs is inflated by fixed overhead on short clips)
10. **Paired bootstrap** — resampling the same files for both models removes file difficulty from noise, so small real differences show up

---

## Results Summary

### Accuracy
- Best clean WER: **3.66%** (medium, 16 kHz) on 200 LibriSpeech files
- Small model (4.21%) is not reliably worse than medium on this test set
- Telephone bandwidth (8 kHz) doesn't significantly hurt WER; 4 kHz does

### Noise Robustness
- WER stays below 5% at 20 dB for all models
- Sharp degradation starts at 10 dB (base: 11.9%, small: 7.1%, medium: 5.2%)
- At 0 dB, even medium degrades to 16.7%
- Medium model is consistently most robust to noise

### Confidence as Quality Proxy
- avg_logprob has strong negative correlation with WER: **ρ = -0.605** pooled, **-0.800** at 0 dB
- This enables quality monitoring without reference text — the decoder's own confidence predicts errors
- no_speech_prob is a moderate secondary signal (ρ = 0.272 pooled)

### Throughput
- tiny/base: ~10x real-time (RTF ≈ 0.06)
- small: ~12x real-time (RTF ≈ 0.08)
- medium: ~7x real-time (RTF ≈ 0.14)
- All measured on RTX 3050 (4 GB VRAM) with int8_float16 quantization
