"""Chooses the model implementation. Heavy imports happen lazily so the API process
(which never runs a model) stays light."""
from functools import lru_cache

from app.config import get_settings
from app.models.base import ModelWrapper
from app.models.stub import StubModel

DEFAULTS = {
    "model_size": None,            # None -> use WHISPER_MODEL_SIZE from the environment
    "language": None,              # None -> auto-detect
    "beam_size": 5,
    "simulate_sample_rate": None,  # e.g. 8000 to mimic telephone audio
    "max_chunk_seconds": None,     # None -> settings.max_chunk_seconds
}
ALLOWED_KEYS = set(DEFAULTS)


def describe_model(config: dict) -> str:
    """Model version string stored on every run (no model is loaded)."""
    s = get_settings()
    if s.model_backend == "stub":
        return "stub-v1"
    size = config.get("model_size") or s.whisper_model_size
    return f"faster-whisper-{size}-{s.whisper_compute_type}"


@lru_cache(maxsize=1)
def _stub() -> StubModel:
    s = get_settings()
    return StubModel(s.stub_sleep_seconds, s.stub_sleep_per_audio_second, s.stub_failure_probability)


def get_model(config: dict | None = None) -> ModelWrapper:
    s = get_settings()
    config = config or {}
    if s.model_backend == "stub":
        return _stub()
    if s.model_backend == "whisper":
        from app.models.whisper_model import get_whisper
        size = config.get("model_size") or s.whisper_model_size
        return get_whisper(size, s.whisper_device, s.whisper_compute_type)
    raise ValueError(f"unknown MODEL_BACKEND: {s.model_backend}")
