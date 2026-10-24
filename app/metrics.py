"""Prometheus metrics derived from PostgreSQL at scrape time.
Why from the DB: the DB is the source of truth, and it works the same no matter how many
API/worker processes exist (no per-process registries to merge)."""
import math
from datetime import datetime, timedelta, timezone

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import repositories as repo
from app.db.models import RunStatus

registry = CollectorRegistry()
QUEUE_DEPTH = Gauge("asr_queue_depth", "Messages waiting in the Redis queue", registry=registry)
RUNS_BY_STATUS = Gauge("asr_runs_by_status", "All runs by status", ["status"], registry=registry)
WINDOW_RUNS = Gauge("asr_window_runs", "Runs finished in the recent window", ["status"], registry=registry)
WINDOW_STAT = Gauge("asr_window_stat", "Statistics over runs finished in the recent window",
                    ["metric", "stat"], registry=registry)

SIGNALS = {
    "queue_wait_s": lambda r: (r.started_at - r.queued_at).total_seconds() if r.started_at else None,
    "processing_s": lambda r: (r.finished_at - r.started_at).total_seconds()
                              if r.started_at and r.finished_at else None,
    "rtf": lambda r: r.rtf,
    "avg_logprob": lambda r: r.avg_logprob,
    "no_speech_prob": lambda r: r.no_speech_prob,
    "input_rms": lambda r: r.input_rms,
    "silence_ratio": lambda r: r.silence_ratio,
}


def _pct(sorted_vals: list[float], q: float) -> float:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def update_metrics(db: Session) -> None:
    from app.services.run_service import queue_length
    s = get_settings()

    try:
        QUEUE_DEPTH.set(queue_length())
    except Exception:
        QUEUE_DEPTH.set(float("nan"))

    counts = repo.status_counts(db)
    for st in RunStatus:
        RUNS_BY_STATUS.labels(st.value).set(counts.get(st.value, 0))

    since = datetime.now(timezone.utc) - timedelta(seconds=s.metrics_window_seconds)
    runs = repo.finished_runs_since(db, since)
    for st in (RunStatus.done, RunStatus.failed):
        WINDOW_RUNS.labels(st.value).set(sum(1 for r in runs if r.status == st))
    done = [r for r in runs if r.status == RunStatus.done]
    for name, fn in SIGNALS.items():
        vals = sorted(v for v in (fn(r) for r in done) if v is not None)
        if vals:
            WINDOW_STAT.labels(name, "mean").set(sum(vals) / len(vals))
            WINDOW_STAT.labels(name, "p50").set(_pct(vals, 0.50))
            WINDOW_STAT.labels(name, "p95").set(_pct(vals, 0.95))
        else:
            for stat in ("mean", "p50", "p95"):
                WINDOW_STAT.labels(name, stat).set(math.nan)


def render() -> tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST
