from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db import repositories as repo
from app.db.models import User
from app.db.session import get_db
from app.services import audio_service
from app.services.errors import NotFound
from app.storage import get_storage

router = APIRouter(prefix="/audio", tags=["audio"])


def audio_to_dict(a) -> dict:
    return {"id": a.id, "filename": a.original_filename, "duration_seconds": a.duration_seconds,
            "sample_rate": a.sample_rate, "channels": a.channels, "size_bytes": a.size_bytes,
            "checksum": a.checksum, "dataset_tag": a.dataset_tag}


@router.post("", status_code=201)
def upload_audio(file: UploadFile = File(...), dataset_tag: str | None = Form(None),
                 reference_text: str | None = Form(None),
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    audio, duplicate = audio_service.submit_audio(
        db, get_storage(), user, file.file, file.filename or "", dataset_tag, reference_text)
    return {**audio_to_dict(audio), "duplicate": duplicate}


@router.get("/{audio_id}")
def get_audio(audio_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    a = repo.get_audio(db, audio_id)
    if a is None or a.user_id != user.id:
        raise NotFound("audio not found")
    return audio_to_dict(a)
