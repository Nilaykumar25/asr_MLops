# Speech MLOps: Whisper ASR backend with time-series monitoring

Asynchronous ASR backend (FastAPI + Celery + Redis + PostgreSQL) that preserves original audio,
runs transcription as background jobs, survives failures, and exposes time-series metrics
(Prometheus/Grafana) plus drift detection. Full walkthrough: see the Word guide.

Quick start (WSL2/Linux, Docker, Python 3.10+, ffmpeg):

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env            # edit DATA_ROOT
    docker compose up -d            # PostgreSQL + Redis
    alembic revision --autogenerate -m "initial schema" && alembic upgrade head
    python -m scripts.create_user alice
    uvicorn app.api.main:app --port 8000                                   # terminal 2
    celery -A app.worker.celery_app:celery_app worker -B -l info -c 4      # terminal 3
    pytest -q                       # unit + API tests (no Docker needed)
