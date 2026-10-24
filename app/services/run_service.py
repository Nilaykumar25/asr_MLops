import hashlib
import json
import logging
from datetime import datetime, timezone

import redis
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import repositories as repo
from app.db.models import Run, RunStatus, User
from app.models.registry import ALLOWED_KEYS, DEFAULTS, describe_model
from app.services.errors import BadRequest, NotFound, ServiceUnavailable, TooManyRequests

log = logging.getLogger("asr.runs")
_redis = redis.Redis.from_url(get_settings().redis_url)


def queue_length() -> int:
    """Number of messages waiting in Celery's default Redis list."""
    return int(_redis.llen("celery"))


def resolve_config(user_config: dict | None) -> dict:
    cfg = dict(user_config or {})
    unknown = set(cfg) - ALLOWED_KEYS
    if unknown:
        raise BadRequest(f"unknown config keys: {sorted(unknown)}. Allowed: {sorted(ALLOWED_KEYS)}")
    resolved = {**DEFAULTS, **cfg}
    sr = resolved["simulate_sample_rate"]
    if sr is not None and not (1000 <= int(sr) <= 48000):
        raise BadRequest("simulate_sample_rate must be between 1000 and 48000")
    return resolved


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def start_run(db: Session, user: User, audio_id: str, user_config: dict | None) -> Run:
    audio = repo.get_audio(db, audio_id)
    if audio is None or audio.user_id != user.id:
        raise NotFound("audio not found")   # 404, never 403: do not reveal other users' ids

    cfg = resolve_config(user_config)

    # ---- backpressure: per-user concurrency (serialised per user) + global queue length ----
    repo.lock_user(db, user.id)
    if repo.count_active_runs(db, user.id) >= user.max_concurrent_runs:
        db.rollback()
        raise TooManyRequests("too many active runs for this user")
    try:
        depth = queue_length()
    except redis.RedisError:
        db.rollback()
        raise ServiceUnavailable("queue is unavailable, try again shortly")
    if depth >= get_settings().max_queue_length:
        db.rollback()
        raise TooManyRequests("system queue is full, try again later")

    now = datetime.now(timezone.utc)
    run = Run(audio_id=audio.id, user_id=user.id, model_version=describe_model(cfg), config=cfg,
              config_hash=config_hash(cfg), status=RunStatus.queued, queued_at=now, last_enqueued_at=now)
    db.add(run)
    db.commit()

    from app.worker.tasks import process_run
    try:
        process_run.delay(run.id)
    except Exception:  # broker hiccup: the row is 'queued', the reaper will re-enqueue it
        log.exception("run_id=%s enqueue failed; reaper will retry", run.id)
    return run


def get_run_for_user(db: Session, user: User, run_id: str) -> Run:
    run = repo.get_run(db, run_id)
    if run is None or run.user_id != user.id:
        raise NotFound("run not found")
    return run
