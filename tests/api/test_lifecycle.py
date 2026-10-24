import io

from tests.conftest import make_wav


def upload(client, headers, seed=0, seconds=1.0, name="a.wav"):
    return client.post("/audio", headers=headers,
                       files={"file": (name, io.BytesIO(make_wav(seconds, seed=seed)), "audio/wav")})


def test_auth_required(client):
    assert client.post("/audio").status_code in (401, 422)
    assert client.get("/runs/x", headers={"X-API-Key": "wrong"}).status_code == 401


def test_full_lifecycle(client, make_user):
    h = make_user()
    r = upload(client, h, seed=1, seconds=2.0)
    assert r.status_code == 201, r.text
    audio = r.json()
    assert abs(audio["duration_seconds"] - 2.0) < 0.1 and audio["duplicate"] is False

    r = client.post(f"/audio/{audio['id']}/transcribe", headers=h, json={"config": {}})
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    run = client.get(f"/runs/{run_id}", headers=h).json()   # eager mode: already finished
    assert run["status"] == "done"
    assert run["transcript"].startswith("stub text")
    assert run["timings"]["total_s"] is not None and run["attempts"] == 1


def test_duplicate_upload_returns_same_record(client, make_user):
    h = make_user()
    a = upload(client, h, seed=5).json()
    b = upload(client, h, seed=5).json()
    assert b["duplicate"] is True and a["id"] == b["id"]


def test_same_file_two_users_is_not_a_duplicate(client, make_user):
    a = upload(client, make_user("u1"), seed=7).json()
    b = upload(client, make_user("u2"), seed=7).json()
    assert a["id"] != b["id"]


def test_rejects_non_audio_and_bad_extension(client, make_user):
    h = make_user()
    r = client.post("/audio", headers=h, files={"file": ("x.wav", io.BytesIO(b"not audio"), "audio/wav")})
    assert r.status_code == 422
    r = client.post("/audio", headers=h, files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/x")})
    assert r.status_code == 415


def test_users_cannot_see_each_others_data(client, make_user):
    alice, bob = make_user("alice"), make_user("bob")
    audio = upload(client, alice, seed=9).json()
    run_id = client.post(f"/audio/{audio['id']}/transcribe", headers=alice, json={}).json()["run_id"]
    assert client.get(f"/audio/{audio['id']}", headers=bob).status_code == 404
    assert client.get(f"/runs/{run_id}", headers=bob).status_code == 404
    assert client.post(f"/audio/{audio['id']}/transcribe", headers=bob, json={}).status_code == 404


def test_unknown_config_key_rejected(client, make_user):
    h = make_user()
    audio = upload(client, h, seed=11).json()
    r = client.post(f"/audio/{audio['id']}/transcribe", headers=h, json={"config": {"bogus": 1}})
    assert r.status_code == 422


def test_per_user_concurrency_limit_returns_429(client, make_user, monkeypatch):
    from app.worker.celery_app import celery_app
    celery_app.conf.task_always_eager = False        # leave runs 'queued' so they stay active
    monkeypatch.setattr("app.worker.tasks.process_run.delay", lambda *_: None)
    try:
        h = make_user("limited", max_runs=2)
        audio = upload(client, h, seed=13).json()
        codes = [client.post(f"/audio/{audio['id']}/transcribe", headers=h, json={}).status_code
                 for _ in range(3)]
        assert codes == [202, 202, 429]
    finally:
        celery_app.conf.task_always_eager = True


def test_queue_full_returns_429(client, make_user, monkeypatch):
    monkeypatch.setattr("app.services.run_service.queue_length", lambda: 10**9)
    h = make_user()
    audio = upload(client, h, seed=15).json()
    assert client.post(f"/audio/{audio['id']}/transcribe", headers=h, json={}).status_code == 429


def test_metrics_endpoint(client, make_user):
    h = make_user()
    audio = upload(client, h, seed=17).json()
    client.post(f"/audio/{audio['id']}/transcribe", headers=h, json={})
    text = client.get("/metrics").text
    assert 'asr_runs_by_status{status="done"} 1.0' in text


def test_redis_down_returns_503_not_500(client, make_user, monkeypatch):
    import redis

    def boom():
        raise redis.ConnectionError("redis is down")

    monkeypatch.setattr("app.services.run_service.queue_length", boom)
    h = make_user()
    audio = upload(client, h, seed=19).json()
    assert client.post(f"/audio/{audio['id']}/transcribe", headers=h, json={}).status_code == 503
