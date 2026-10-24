"""All SQL lives here. Services and workers call these functions; they never write queries."""
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import ACTIVE, Audio, Chunk, Run, RunStatus, User


# ---- users -------------------------------------------------------------------
def get_user_by_key_hash(db: Session, key_hash: str) -> User | None:
    return db.scalar(select(User).where(User.api_key_hash == key_hash))


def lock_user(db: Session, user_id: str) -> None:
    """Serialises per-user checks on Postgres (SELECT ... FOR UPDATE). No-op on SQLite."""
    db.execute(select(User.id).where(User.id == user_id).with_for_update())


# ---- audio -------------------------------------------------------------------
def get_audio(db: Session, audio_id: str) -> Audio | None:
    return db.get(Audio, audio_id)


def get_audio_by_checksum(db: Session, user_id: str, checksum: str) -> Audio | None:
    return db.scalar(select(Audio).where(Audio.user_id == user_id, Audio.checksum == checksum))


# ---- runs --------------------------------------------------------------------
def get_run(db: Session, run_id: str) -> Run | None:
    return db.get(Run, run_id)


def count_active_runs(db: Session, user_id: str) -> int:
    return db.scalar(
        select(func.count()).select_from(Run).where(Run.user_id == user_id, Run.status.in_(ACTIVE))
    ) or 0


def claim_run(db: Session, run_id: str, now: datetime) -> bool:
    """Compare-and-set: only ONE worker can move a run queued -> processing.
    This is what makes duplicate delivery harmless (idempotency)."""
    res = db.execute(
        update(Run)
        .where(Run.id == run_id, Run.status == RunStatus.queued)
        .values(status=RunStatus.processing, started_at=now, attempts=Run.attempts + 1)
    )
    db.commit()
    return res.rowcount == 1


def requeue_run(db: Session, run_id: str, now: datetime) -> None:
    db.execute(
        update(Run)
        .where(Run.id == run_id, Run.status == RunStatus.processing)
        .values(status=RunStatus.queued, last_enqueued_at=now)
    )
    db.commit()


def complete_run(db: Session, run_id: str, now: datetime, *, transcript: str, chunks: list[dict],
                 rtf, avg_logprob, no_speech_prob, input_rms, silence_ratio) -> bool:
    res = db.execute(
        update(Run)
        .where(Run.id == run_id, Run.status == RunStatus.processing)
        .values(status=RunStatus.done, transcript=transcript, finished_at=now, error=None,
                rtf=rtf, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
                input_rms=input_rms, silence_ratio=silence_ratio, n_chunks=len(chunks))
    )
    if res.rowcount == 1:
        db.add_all([Chunk(run_id=run_id, **c) for c in chunks])
    db.commit()
    return res.rowcount == 1


def fail_run(db: Session, run_id: str, now: datetime, error: str) -> None:
    db.execute(
        update(Run)
        .where(Run.id == run_id, Run.status.in_(ACTIVE))
        .values(status=RunStatus.failed, error=error[:2000], finished_at=now)
    )
    db.commit()


def stuck_processing_runs(db: Session, cutoff: datetime) -> list[Run]:
    return list(db.scalars(
        select(Run).where(Run.status == RunStatus.processing, Run.started_at < cutoff)))


def unpicked_queued_runs(db: Session, cutoff: datetime, limit: int = 1000) -> list[Run]:
    return list(db.scalars(
        select(Run).where(Run.status == RunStatus.queued, Run.last_enqueued_at < cutoff).limit(limit)))


def finished_runs_since(db: Session, since: datetime, limit: int = 5000) -> list[Run]:
    return list(db.scalars(
        select(Run).where(Run.finished_at >= since).order_by(Run.finished_at.desc()).limit(limit)))


def status_counts(db: Session) -> dict[str, int]:
    rows = db.execute(select(Run.status, func.count()).group_by(Run.status)).all()
    return {s.value: n for s, n in rows}
