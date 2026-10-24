"""Reliability behaviour: retries, permanent failures, idempotency, reaper."""
import io
from datetime import datetime, timedelta, timezone

from app.db import repositories as repo
from app.db.models import Run, RunStatus
from app.db.session import SessionLocal
from app.worker import tasks
from tests.conftest import make_wav


def make_queued_run(client, headers, seed):
    audio = client.post("/audio", headers=headers,
                        files={"file": ("a.wav", io.BytesIO(make_wav(1, seed=seed)), "audio/wav")}).json()
    from app.worker.celery_app import celery_app
    celery_app.conf.task_always_eager = False
    try:
        import app.worker.tasks as t
        orig = t.process_run.delay
        t.process_run.delay = lambda *_: None
        run_id = client.post(f"/audio/{audio['id']}/transcribe", headers=headers, json={}).json()["run_id"]
        t.process_run.delay = orig
    finally:
        celery_app.conf.task_always_eager = True
    return run_id


def test_second_delivery_is_ignored(client, make_user):
    h = make_user()
    run_id = make_queued_run(client, h, 21)
    assert tasks.process_run.apply(args=[run_id]).get() == "done"
    assert tasks.process_run.apply(args=[run_id]).get() == "skipped"   # duplicate delivery
    with SessionLocal() as db:
        assert repo.get_run(db, run_id).attempts == 1


def test_transient_failure_is_retried_then_succeeds(client, make_user, monkeypatch):
    h = make_user()
    run_id = make_queued_run(client, h, 22)
    calls = {"n": 0}
    from app.models.stub import StubModel
    real = StubModel.transcribe

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("boom")
        return real(self, *a, **kw)

    monkeypatch.setattr(StubModel, "transcribe", flaky)
    tasks.process_run.apply(args=[run_id])
    with SessionLocal() as db:
        run = repo.get_run(db, run_id)
        assert run.status == RunStatus.done and run.attempts == 2


def test_gives_up_after_max_attempts(client, make_user, monkeypatch):
    h = make_user()
    run_id = make_queued_run(client, h, 23)
    from app.models.stub import StubModel
    monkeypatch.setattr(StubModel, "transcribe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("always")))
    tasks.process_run.apply(args=[run_id])
    with SessionLocal() as db:
        run = repo.get_run(db, run_id)
        assert run.status == RunStatus.failed and "always" in run.error
        assert run.attempts == tasks.MAX_ATTEMPTS


def test_reaper_recovers_dead_worker_and_lost_message(client, make_user):
    h = make_user()
    stuck_id = make_queued_run(client, h, 24)
    lost_id = make_queued_run(client, h, 25)
    old = datetime.now(timezone.utc) - timedelta(hours=5)
    with SessionLocal() as db:
        s = repo.get_run(db, stuck_id)       # simulate: worker died mid-run long ago
        s.status, s.started_at, s.attempts = RunStatus.processing, old, 1
        l = repo.get_run(db, lost_id)        # simulate: broker message vanished
        l.last_enqueued_at = old
        db.commit()
    assert tasks.reap_stuck_runs.apply().get() == 2    # eager: re-enqueued runs execute immediately
    with SessionLocal() as db:
        assert repo.get_run(db, stuck_id).status == RunStatus.done
        assert repo.get_run(db, lost_id).status == RunStatus.done
