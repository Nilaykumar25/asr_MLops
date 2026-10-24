import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, BigInteger, DateTime, Enum, Float, ForeignKey, Index, Integer,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


TERMINAL = (RunStatus.done, RunStatus.failed)
ACTIVE = (RunStatus.queued, RunStatus.processing)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    max_concurrent_runs: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Audio(Base):
    __tablename__ = "audio"
    __table_args__ = (UniqueConstraint("user_id", "checksum", name="uq_audio_user_checksum"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    checksum: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    duration_seconds: Mapped[float] = mapped_column(Float)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    channels: Mapped[int | None] = mapped_column(Integer)
    # research fields (used from Phase 5 onward)
    dataset_tag: Mapped[str | None] = mapped_column(String(64), index=True)
    reference_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Run(Base):
    __tablename__ = "transcription_runs"
    __table_args__ = (Index("ix_runs_status_queued", "status", "queued_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    audio_id: Mapped[str] = mapped_column(ForeignKey("audio.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(128))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    config_hash: Mapped[str] = mapped_column(String(40))
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=20), default=RunStatus.queued)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    transcript: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    # timing (these produce queue wait, processing time, total latency)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_enqueued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # per-run monitoring signals (filled in by the worker; these feed Prometheus + drift)
    rtf: Mapped[float | None] = mapped_column(Float)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)
    input_rms: Mapped[float | None] = mapped_column(Float)
    silence_ratio: Mapped[float | None] = mapped_column(Float)
    n_chunks: Mapped[int | None] = mapped_column(Integer)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("transcription_runs.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    avg_logprob: Mapped[float | None] = mapped_column(Float)
    no_speech_prob: Mapped[float | None] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="chunks")
