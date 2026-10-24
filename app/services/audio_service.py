import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import repositories as repo
from app.db.models import Audio, User, new_id
from app.preprocessing import audio_io
from app.services.errors import BadRequest, PayloadTooLarge, UnsupportedMedia
from app.storage.base import StorageBackend

log = logging.getLogger("asr.audio")


class _LimitedStream:
    """Aborts an upload as soon as it exceeds the size limit (no need to store it all first)."""

    def __init__(self, stream: BinaryIO, limit: int):
        self._stream, self._limit, self._read = stream, limit, 0

    def read(self, n: int = -1) -> bytes:
        buf = self._stream.read(n)
        self._read += len(buf)
        if self._read > self._limit:
            raise PayloadTooLarge(f"file exceeds {self._limit // (1024 * 1024)} MB limit")
        return buf


def submit_audio(db: Session, storage: StorageBackend, user: User, stream: BinaryIO, filename: str,
                 dataset_tag: str | None = None, reference_text: str | None = None) -> tuple[Audio, bool]:
    """Validate, store and register an upload. Returns (audio, is_duplicate)."""
    s = get_settings()
    ext = Path(filename or "").suffix.lstrip(".").lower()
    if ext not in s.extensions:
        raise UnsupportedMedia(f"unsupported file type '.{ext}'. Allowed: {sorted(s.extensions)}")

    key = f"audio/{datetime.now(timezone.utc):%Y-%m-%d}/{new_id()}.{ext}"
    try:
        size, checksum = storage.save(key, _LimitedStream(stream, s.max_upload_mb * 1024 * 1024))
    except PayloadTooLarge:
        storage.delete(key)  # nothing was committed; remove partial object if any
        raise

    # duplicate upload by the same user -> return the existing record, discard the new blob
    existing = repo.get_audio_by_checksum(db, user.id, checksum)
    if existing:
        storage.delete(key)
        return existing, True

    # validate: must be decodable and not too long
    local = storage.local_path(key)
    tmpdir = None
    try:
        if local is None:
            tmpdir = tempfile.TemporaryDirectory(prefix="probe_")
            local = storage.fetch_to_path(key, Path(tmpdir.name) / f"in.{ext}")
        info = audio_io.probe(local)
    except audio_io.AudioError as e:
        storage.delete(key)
        # the detailed reason (it contains server paths) goes to the log, never to the client
        log.warning("rejected upload filename=%r user=%s reason=%s", filename, user.username, e)
        raise BadRequest("file is not valid audio or is corrupted")
    finally:
        if tmpdir:
            tmpdir.cleanup()
    if info["duration"] > s.max_duration_seconds:
        storage.delete(key)
        raise BadRequest(f"audio longer than {s.max_duration_seconds} seconds")

    audio = Audio(user_id=user.id, original_filename=filename, storage_key=key, checksum=checksum,
                  size_bytes=size, duration_seconds=info["duration"], sample_rate=info["sample_rate"],
                  channels=info["channels"], dataset_tag=dataset_tag, reference_text=reference_text)
    db.add(audio)
    try:
        db.commit()
    except IntegrityError:  # two simultaneous identical uploads: the other one won the race
        db.rollback()
        storage.delete(key)
        existing = repo.get_audio_by_checksum(db, user.id, checksum)
        if existing:
            return existing, True
        raise
    return audio, False