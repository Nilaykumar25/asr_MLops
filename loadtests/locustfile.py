"""Load test.
  Submit-only (fast, reaches 10,000+ requests):  locust -f loadtests/locustfile.py SubmitterUser
  Full cycle (upload -> transcribe -> wait done): locust -f loadtests/locustfile.py FullCycleUser
Create keys first:  python -m scripts.create_user --count 50 --prefix load --keys-file loadtests/keys.txt
Every uploaded file is unique (random samples), so the dedupe rule never short-circuits storage."""
import io
import itertools
import os
import random
import time
import wave

from locust import HttpUser, between, task

KEYS_FILE = os.environ.get("KEYS_FILE", "loadtests/keys.txt")
KEYS = [k.strip() for k in open(KEYS_FILE) if k.strip()]
_next_key = itertools.cycle(KEYS)
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = float(os.environ.get("POLL_TIMEOUT_S", 900))


def random_wav(seconds: float, sr: int = 16000) -> bytes:
    """Valid mono 16-bit wav filled with random noise: unique checksum on every call."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(os.urandom(int(seconds * sr) * 2))
    return buf.getvalue()


class _Base(HttpUser):
    abstract = True
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.headers = {"X-API-Key": next(_next_key)}

    def _submit(self):
        wav = random_wav(random.uniform(1, 5))
        with self.client.post("/audio", headers=self.headers, name="POST /audio", catch_response=True,
                              files={"file": (f"load{random.randrange(10**9)}.wav", wav, "audio/wav")}) as r:
            if r.status_code != 201:
                r.failure(f"upload {r.status_code}: {r.text[:100]}")
                return None
            audio_id = r.json()["id"]
        with self.client.post(f"/audio/{audio_id}/transcribe", headers=self.headers, json={},
                              name="POST /audio/[id]/transcribe", catch_response=True) as r:
            if r.status_code == 429:
                r.success()           # backpressure working as designed; counted separately below
                self.environment.events.request.fire(
                    request_type="BACKPRESSURE", name="429 rejected", response_time=0,
                    response_length=0, exception=None, context={})
                return None
            if r.status_code != 202:
                r.failure(f"transcribe {r.status_code}: {r.text[:100]}")
                return None
            return r.json()["run_id"]


class SubmitterUser(_Base):
    """Fire and forget. Measures how fast the API can accept work."""

    @task
    def submit(self):
        self._submit()


class FullCycleUser(_Base):
    """Waits for each run to finish. Measures end-to-end latency as a user would see it."""

    @task
    def full_cycle(self):
        t0 = time.time()
        run_id = self._submit()
        if not run_id:
            return
        status = "timeout"
        while time.time() - t0 < POLL_TIMEOUT_S:
            with self.client.get(f"/runs/{run_id}", headers=self.headers, name="GET /runs/[id]",
                                 catch_response=True) as r:
                status = r.json().get("status") if r.status_code == 200 else "error"
            if status in ("done", "failed"):
                break
            time.sleep(POLL_INTERVAL_S)
        self.environment.events.request.fire(
            request_type="E2E", name="upload -> terminal state",
            response_time=(time.time() - t0) * 1000, response_length=0,
            exception=None if status == "done" else Exception(f"ended as {status}"), context={})
