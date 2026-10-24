import logging
import time
from datetime import datetime, timedelta, timezone

from celery.signals import worker_process_init

from app.config import get_settings
from app.db import repositories as repo
from app.db.models import Run
from app.db.session import engine, session_scope
from app.models.registry import get_model
from app.services.errors import PermanentRunError
from app.storage import get_storage
from app.worker.celery_app import celery_app

log = logging.getLogger("asr.worker")
settings = get_settings()
MAX_ATTEMPTS = settings.task_max_retries + 1


@worker_process_init.connect
def _fresh_db_connections(**_):
    engine.dispose()  # forked children must not share the parent's pooled connections


def _now() -> datetime:
    return datetime.now(timezone.utc)


@celery_app.task(name="app.worker.tasks.process_run", bind=True)
def process_run(self, run_id: str):
    from app.pipeline import run_pipeline  # lazy: heavy imports only in the worker

    # 1. claim (compare-and-set). Duplicate deliveries lose here and exit quietly.
    with session_scope() as db:
        if not repo.claim_run(db, run_id, _now()):
            log.info("run_id=%s skipped (not in queued state)", run_id)
            return "skipped"
        run = repo.get_run(db, run_id)
        audio = repo.get_audio(db, run.audio_id)
    log.info("run_id=%s started attempt=%d", run_id, run.attempts)

    # 2. do the work
    try:
        t0 = time.perf_counter()
        out = run_pipeline(run, audio, get_storage(), get_model(run.config))
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        return _handle_failure(self, run_id, exc)

    # 3. record the result
    res = out.result
    lp = [c.avg_logprob for c in res.chunks if c.avg_logprob is not None]
    ns = [c.no_speech_prob for c in res.chunks if c.no_speech_prob is not None]
    chunk_rows = [dict(idx=i, start_s=c.start, end_s=c.end, text=c.text,
                       avg_logprob=c.avg_logprob, no_speech_prob=c.no_speech_prob)
                  for i, c in enumerate(res.chunks)]
    with session_scope() as db:
        repo.complete_run(
            db, run_id, _now(), transcript=res.text, chunks=chunk_rows,
            rtf=(elapsed / audio.duration_seconds) if audio.duration_seconds else None,
            avg_logprob=sum(lp) / len(lp) if lp else None,
            no_speech_prob=sum(ns) / len(ns) if ns else None,
            input_rms=out.input_rms, silence_ratio=out.silence_ratio)
    log.info("run_id=%s done in %.2fs", run_id, elapsed)
    return "done"


def _handle_failure(task, run_id: str, exc: Exception):
    permanent = isinstance(exc, PermanentRunError)
    with session_scope() as db:
        run = repo.get_run(db, run_id)
        attempts = run.attempts
        if permanent or attempts >= MAX_ATTEMPTS:
            repo.fail_run(db, run_id, _now(), f"{type(exc).__name__}: {exc}")
            log.error("run_id=%s FAILED after %d attempt(s): %s", run_id, attempts, exc)
            return "failed"
        repo.requeue_run(db, run_id, _now())
    countdown = min(2 ** attempts, 60)  # 2s, 4s, 8s ... capped at 60s
    log.warning("run_id=%s attempt %d failed (%s); retrying in %ds", run_id, attempts, exc, countdown)
    raise task.retry(exc=exc, countdown=countdown, max_retries=settings.task_max_retries)


@celery_app.task(name="app.worker.tasks.reap_stuck_runs")
def reap_stuck_runs():
    """Safety net. Re-enqueues runs that fell through the cracks:
    (a) 'processing' for too long -> its worker probably died;
    (b) 'queued' but the broker message was lost (Redis restart, enqueue failure).
    Duplicates are harmless because claim_run is a compare-and-set."""
    now = _now()
    to_enqueue: set[str] = set()
    with session_scope() as db:
        for r in repo.stuck_processing_runs(db, now - timedelta(seconds=settings.stuck_run_timeout_seconds)):
            if r.attempts >= MAX_ATTEMPTS:
                repo.fail_run(db, r.id, now, "worker lost / timed out too many times")
            else:
                repo.requeue_run(db, r.id, now)
                to_enqueue.add(r.id)
        for r in repo.unpicked_queued_runs(db, now - timedelta(seconds=settings.requeue_after_seconds)):
            r.last_enqueued_at = now
            to_enqueue.add(r.id)
    for rid in to_enqueue:
        process_run.delay(rid)
    if to_enqueue:
        log.warning("reaper re-enqueued %d run(s)", len(to_enqueue))
    return len(to_enqueue)
