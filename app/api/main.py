import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import audio, health, runs
from app.services.errors import ServiceError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="ASR MLOps Backend", version="1.0")
app.include_router(health.router)
app.include_router(audio.router)
app.include_router(runs.router)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
