"""All configuration lives here. Nothing else in the codebase reads os.environ."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Phase 0 constants -------------------------------------------------
    target_sample_rate: int = 16000          # inference sample rate (mono)
    max_chunk_seconds: float = 30.0          # forced-cut fallback for VAD chunking
    allowed_extensions: str = "wav,flac,mp3,ogg,m4a"

    # ---- storage -------------------------------------------------------------
    data_root: str = "./data"
    storage_backend: str = "local"           # local | s3
    s3_endpoint_url: str | None = None
    s3_bucket: str = "asr-audio"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"

    # ---- database / queue ----------------------------------------------------
    database_url: str = "sqlite:///./dev.db"
    db_pool_size: int = 10
    db_max_overflow: int = 10
    redis_url: str = "redis://localhost:6379/0"

    # ---- upload validation ---------------------------------------------------
    max_upload_mb: int = 500
    max_duration_seconds: int = 10800        # 3 hours

    # ---- backpressure --------------------------------------------------------
    max_queue_length: int = 50000
    default_max_concurrent_runs: int = 100

    # ---- worker / reliability ------------------------------------------------
    worker_concurrency: int = 4
    task_max_retries: int = 3                # attempts = retries + 1
    visibility_timeout_seconds: int = 14400  # must exceed your longest task
    stuck_run_timeout_seconds: int = 3600    # 'processing' longer than this is reaped
    requeue_after_seconds: int = 3600        # 'queued' but never picked up is re-enqueued
    metrics_window_seconds: int = 300

    # ---- model ---------------------------------------------------------------
    model_backend: str = "stub"              # stub | whisper
    stub_sleep_seconds: float = 0.5
    stub_sleep_per_audio_second: float = 0.0
    stub_failure_probability: float = 0.05
    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    vad_backend: str = "silero"              # silero | none

    @property
    def extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
