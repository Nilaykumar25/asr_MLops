"""Voice Activity Detection behind an interface, so the VAD can be swapped (Silero today)."""
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import get_settings


class VADBackend(ABC):
    name: str

    @abstractmethod
    def detect_speech_segments(self, audio, sample_rate: int) -> list[tuple[float, float]]:
        """Return [(start_s, end_s), ...] of speech regions."""


class NoVAD(VADBackend):
    """Treats the whole file as speech. Chunking then falls back to fixed 30 s windows."""
    name = "none"

    def detect_speech_segments(self, audio, sample_rate):
        return [(0.0, len(audio) / sample_rate)] if len(audio) else []


class SileroVAD(VADBackend):
    name = "silero"

    def __init__(self):
        from silero_vad import load_silero_vad
        self._model = load_silero_vad()

    def detect_speech_segments(self, audio, sample_rate):
        import torch
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(
            torch.from_numpy(audio), self._model, sampling_rate=sample_rate,
            return_seconds=True, min_silence_duration_ms=300)
        return [(float(t["start"]), float(t["end"])) for t in ts]


@lru_cache
def get_vad() -> VADBackend:
    backend = get_settings().vad_backend
    if backend == "silero":
        return SileroVAD()
    if backend == "none":
        return NoVAD()
    raise ValueError(f"unknown VAD_BACKEND: {backend}")
