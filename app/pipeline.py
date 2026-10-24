"""PHASE 4 VERSION (replaces the Phase 2 placeholder).
original audio -> ffmpeg -> 16 kHz mono wav (cached) -> VAD -> chunks -> model."""
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.models.base import ModelWrapper, TranscriptionResult
from app.preprocessing import audio_io
from app.preprocessing.chunking import build_chunks
from app.services.errors import PermanentRunError

log = logging.getLogger("asr.pipeline")


@dataclass
class PipelineOutput:
    result: TranscriptionResult
    input_rms: float | None = None
    silence_ratio: float | None = None


def derived_key(audio_id: str, config: dict) -> str:
    """Cache key of the resampled wav. The recipe is part of the name, so changing the
    recipe can never serve a stale file."""
    s = get_settings()
    tag = f"{s.target_sample_rate // 1000}k_mono"
    if config.get("simulate_sample_rate"):
        tag += f"_via{int(config['simulate_sample_rate'])}hz"
    return f"derived/{audio_id}/{tag}.wav"


def _prepare_wav(audio, config: dict, storage, tmp: Path) -> Path:
    """Return a local path to the 16 kHz mono wav, creating + caching it if needed."""
    s = get_settings()
    key = derived_key(audio.id, config)
    wav = tmp / "audio_16k.wav"
    if storage.exists(key):
        return storage.fetch_to_path(key, wav)
    ext = Path(audio.storage_key).suffix
    src = storage.fetch_to_path(audio.storage_key, tmp / f"original{ext}")
    try:
        audio_io.resample_to_wav(src, wav, s.target_sample_rate, config.get("simulate_sample_rate"))
    except audio_io.AudioError as e:
        raise PermanentRunError(f"cannot decode audio: {e}")   # retrying will not help
    try:
        with open(wav, "rb") as f:
            storage.save(key, f)
    except FileExistsError:
        pass  # another worker cached it first: fine
    return wav


def run_pipeline(run, audio, storage, model: ModelWrapper) -> PipelineOutput:
    s = get_settings()
    config = run.config or {}
    sr = s.target_sample_rate

    # The stub does not need audio: skip decode/VAD (keeps load tests cheap).
    if not model.needs_audio:
        chunks = [(0.0, float(audio.duration_seconds or 0.0))]
        return PipelineOutput(model.transcribe(None, sr, chunks, config))

    import numpy as np
    from app.preprocessing.vad import get_vad

    with tempfile.TemporaryDirectory(prefix="asr_") as tmp:
        data = audio_io.load_wav(_prepare_wav(audio, config, storage, Path(tmp)))

    total = len(data) / sr
    vad = get_vad()
    segments = vad.detect_speech_segments(data, sr)
    speech = sum(e - s_ for s_, e in segments)
    silence_ratio = (1.0 - speech / total) if total > 0 else None
    rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0.0

    max_len = float(config.get("max_chunk_seconds") or s.max_chunk_seconds)
    chunks = build_chunks(segments, max_len)
    log.info("run_id=%s duration=%.1fs speech=%.1fs chunks=%d vad=%s",
             run.id, total, speech, len(chunks), vad.name)
    if not chunks:   # pure silence: valid result, empty transcript
        return PipelineOutput(TranscriptionResult("", []), rms, silence_ratio)
    return PipelineOutput(model.transcribe(data, sr, chunks, config), rms, silence_ratio)
