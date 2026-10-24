from celery import Celery

from app.config import get_settings

s = get_settings()

celery_app = Celery("asr", broker=s.redis_url, backend=None, include=["app.worker.tasks"])
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,                 # PostgreSQL is the source of truth, not a result backend
    task_acks_late=True,                     # ack AFTER the task finishes -> a crashed worker's job is redelivered
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,            # one job at a time per process: fair scheduling for long jobs
    worker_concurrency=s.worker_concurrency,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": s.visibility_timeout_seconds},
    timezone="UTC",
    beat_schedule={
        "reap-stuck-runs": {"task": "app.worker.tasks.reap_stuck_runs", "schedule": 60.0},
    },
)
