from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.services import run_service

router = APIRouter(tags=["runs"])


class TranscribeRequest(BaseModel):
    config: dict | None = None


def _secs(a, b):
    return None if a is None or b is None else round((b - a).total_seconds(), 4)


def run_to_dict(r) -> dict:
    return {
        "id": r.id, "audio_id": r.audio_id, "status": r.status.value,
        "model_version": r.model_version, "config": r.config, "attempts": r.attempts,
        "transcript": r.transcript, "error": r.error,
        "timings": {"queue_wait_s": _secs(r.queued_at, r.started_at),
                    "processing_s": _secs(r.started_at, r.finished_at),
                    "total_s": _secs(r.queued_at, r.finished_at)},
        "signals": {"rtf": r.rtf, "avg_logprob": r.avg_logprob, "no_speech_prob": r.no_speech_prob,
                    "input_rms": r.input_rms, "silence_ratio": r.silence_ratio, "n_chunks": r.n_chunks},
    }


@router.post("/audio/{audio_id}/transcribe", status_code=202)
def start_transcription(audio_id: str, body: TranscribeRequest | None = None,
                        user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    run = run_service.start_run(db, user, audio_id, body.config if body else None)
    return {"run_id": run.id, "status": run.status.value}


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return run_to_dict(run_service.get_run_for_user(db, user, run_id))
