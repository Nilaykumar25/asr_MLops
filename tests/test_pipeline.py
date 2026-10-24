"""Tests the real Phase 4 pipeline using a fake audio-consuming model (no Whisper download needed)."""
import io
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("soundfile")

from app import pipeline  # noqa: E402
from app.models.base import ChunkResult, ModelWrapper, TranscriptionResult  # noqa: E402
from app.preprocessing import vad as vadmod  # noqa: E402
from app.storage.local import LocalStorage  # noqa: E402
from tests.conftest import make_wav  # noqa: E402


class FakeAudioModel(ModelWrapper):
    needs_audio = True
    version = "fake"

    def __init__(self):
        self.seen = None

    def transcribe(self, audio, sample_rate, chunks, config):
        self.seen = (len(audio), sample_rate, chunks)
        return TranscriptionResult("hello world", [ChunkResult(s, e, "hello", -0.2, 0.01) for s, e in chunks])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(vadmod, "get_vad", lambda: vadmod.NoVAD())
    storage = LocalStorage(str(tmp_path))
    storage.save("audio/a.wav", io.BytesIO(make_wav(70, sr=8000, seed=1)))   # 70 s of 8 kHz audio
    audio = SimpleNamespace(id="aud1", storage_key="audio/a.wav", duration_seconds=70.0)
    return storage, audio


def test_resamples_to_16k_chunks_and_caches(setup):
    storage, audio = setup
    model = FakeAudioModel()
    run = SimpleNamespace(id="r1", config={})
    out = pipeline.run_pipeline(run, audio, storage, model)

    n_samples, sr, chunks = model.seen
    assert sr == 16000 and abs(n_samples / sr - 70) < 0.5          # resampled 8k -> 16k
    assert chunks[0] == (0.0, 30.0) and all(e - s <= 30.0 for s, e in chunks)
    assert out.input_rms > 0 and out.silence_ratio == pytest.approx(0.0, abs=0.01)
    assert storage.exists("derived/aud1/16k_mono.wav")             # derived copy cached

    # second run reuses the cached derived file, originals untouched
    pipeline.run_pipeline(run, audio, storage, FakeAudioModel())
    assert storage.exists("audio/a.wav")


def test_simulated_low_bandwidth_gets_its_own_cache_key(setup):
    storage, audio = setup
    run = SimpleNamespace(id="r2", config={"simulate_sample_rate": 4000})
    pipeline.run_pipeline(run, audio, storage, FakeAudioModel())
    assert storage.exists("derived/aud1/16k_mono_via4000hz.wav")


def test_undecodable_audio_is_a_permanent_error(tmp_path, monkeypatch):
    from app.services.errors import PermanentRunError
    monkeypatch.setattr(vadmod, "get_vad", lambda: vadmod.NoVAD())
    storage = LocalStorage(str(tmp_path))
    storage.save("audio/bad.wav", io.BytesIO(b"garbage" * 100))
    audio = SimpleNamespace(id="bad", storage_key="audio/bad.wav", duration_seconds=1.0)
    with pytest.raises(PermanentRunError):
        pipeline.run_pipeline(SimpleNamespace(id="r3", config={}), audio, storage, FakeAudioModel())


def test_stub_path_skips_audio_processing(setup):
    from app.models.stub import StubModel
    storage, audio = setup
    out = pipeline.run_pipeline(SimpleNamespace(id="r4", config={}), audio, storage, StubModel(0, 0, 0))
    assert out.result.text.startswith("stub text")
    assert not storage.exists("derived/aud1/16k_mono.wav")
