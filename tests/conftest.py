"""Test environment. Variables are set BEFORE the app is imported, so the app uses a throw-away
SQLite database and a temp storage folder. No Postgres or Redis is needed for unit tests."""
import io
import os
import struct
import tempfile
import wave

_tmp = tempfile.mkdtemp(prefix="asr_test_")
os.environ.update({
    "DATABASE_URL": f"sqlite:///{_tmp}/test.db",
    "DATA_ROOT": f"{_tmp}/data",
    "STORAGE_BACKEND": "local",
    "MODEL_BACKEND": "stub",
    "STUB_SLEEP_SECONDS": "0",
    "STUB_FAILURE_PROBABILITY": "0",
    "TASK_MAX_RETRIES": "2",
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.api.deps import hash_key  # noqa: E402
from app.api.main import app  # noqa: E402
from app.db.models import Base, User  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.worker.celery_app import celery_app  # noqa: E402

celery_app.conf.task_always_eager = True          # run tasks inline: no Redis, no worker process


def make_wav(seconds: float = 1.0, sr: int = 16000, seed: int = 0) -> bytes:
    """Valid mono 16-bit wav with deterministic pseudo-random content (so each seed has a unique checksum)."""
    import random
    rnd = random.Random(seed)
    frames = b"".join(struct.pack("<h", rnd.randint(-3000, 3000)) for _ in range(int(sr * seconds)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    monkeypatch.setattr("app.services.run_service.queue_length", lambda: 0)
    yield


@pytest.fixture
def make_user():
    def _make(name="alice", max_runs=100):
        key = f"key-{name}"
        with SessionLocal() as db:
            db.add(User(username=name, api_key_hash=hash_key(key), max_concurrent_runs=max_runs))
            db.commit()
        return {"X-API-Key": key}
    return _make


@pytest.fixture
def client():
    return TestClient(app)
