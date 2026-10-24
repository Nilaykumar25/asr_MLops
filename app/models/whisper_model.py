"""Real ASR model: faster-whisper (CTranslate2). Loaded once per worker process."""
from functools import lru_cache

from app.models.base import ChunkResult, ModelWrapper, TranscriptionResult


class WhisperWrapper(ModelWrapper):
    needs_audio = True

    def __init__(self, size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        self._size, self._compute = size, compute_type
        self._model = WhisperModel(size, device=device, compute_type=compute_type)

    @property
    def version(self) -> str:
        return f"faster-whisper-{self._size}-{self._compute}"

    def transcribe(self, audio, sample_rate, chunks, config) -> TranscriptionResult:
        results: list[ChunkResult] = []
        for start, end in chunks:
            clip = audio[int(start * sample_rate): int(end * sample_rate)]
            if len(clip) < sample_rate * 0.2:      # ignore slivers shorter than 0.2 s
                continue
            segments, _info = self._model.transcribe(
                clip,
                language=config.get("language"),
                beam_size=int(config.get("beam_size") or 5),
                vad_filter=False,                   # we already did VAD ourselves
                condition_on_previous_text=False,   # chunks are independent -> fewer cascading errors
            )
            segs = list(segments)                   # generator -> list (this is where decoding happens)
            text = " ".join(s.text.strip() for s in segs).strip()
            lp = [s.avg_logprob for s in segs]
            ns = [s.no_speech_prob for s in segs]
            results.append(ChunkResult(
                start, end, text,
                avg_logprob=sum(lp) / len(lp) if lp else None,
                no_speech_prob=sum(ns) / len(ns) if ns else None,
            ))
        full = " ".join(c.text for c in results if c.text).strip()
        return TranscriptionResult(full, results)


@lru_cache(maxsize=2)  # keep at most two model sizes in RAM (used by the evaluation grid)
def get_whisper(size: str, device: str, compute_type: str) -> WhisperWrapper:
    return WhisperWrapper(size, device, compute_type)
