import random
import time

from app.models.base import ChunkResult, ModelWrapper, TranscriptionResult


class StubModel(ModelWrapper):
    """Pretends to be an ASR model: sleeps, returns dummy text, fails on demand."""
    needs_audio = False

    def __init__(self, sleep_s: float, per_audio_s: float, failure_p: float):
        self.sleep_s, self.per_audio_s, self.failure_p = sleep_s, per_audio_s, failure_p

    @property
    def version(self) -> str:
        return "stub-v1"

    def transcribe(self, audio, sample_rate, chunks, config) -> TranscriptionResult:
        duration = sum(e - s for s, e in chunks)
        time.sleep(self.sleep_s + self.per_audio_s * duration)
        if random.random() < self.failure_p:
            raise RuntimeError("stub: simulated transient failure")
        results = [ChunkResult(s, e, f"stub text {i}", -0.3, 0.01) for i, (s, e) in enumerate(chunks)]
        return TranscriptionResult(" ".join(c.text for c in results), results)
