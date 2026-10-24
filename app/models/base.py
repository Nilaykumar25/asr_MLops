"""The ModelWrapper contract. The stub and the real Whisper model both implement it,
so the rest of the system never knows which one it is talking to."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class TranscriptionResult:
    text: str
    chunks: list[ChunkResult] = field(default_factory=list)


class ModelWrapper(ABC):
    needs_audio: bool = True   # False => the pipeline may skip decode/resample/VAD (stub)

    @property
    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def transcribe(self, audio, sample_rate: int, chunks: list[tuple[float, float]],
                   config: dict) -> TranscriptionResult:
        """audio: float32 mono numpy array at sample_rate (None when needs_audio is False).
        chunks: list of (start_s, end_s). Returns text + per-chunk results."""
