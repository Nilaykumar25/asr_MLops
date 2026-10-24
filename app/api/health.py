import redis
from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import metrics
from app.config import get_settings
from app.db.session import get_db

router = APIRouter(tags=["ops"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"
    try:
        redis.Redis.from_url(get_settings().redis_url, socket_timeout=2).ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"status": "ok" if ok else "degraded", **checks}, status_code=200 if ok else 503)


@router.get("/metrics")
def prometheus_metrics(db: Session = Depends(get_db)):
    metrics.update_metrics(db)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
