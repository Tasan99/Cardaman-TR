"""Operational and read-only public regulatory endpoints; tenant APIs await OIDC/RBAC."""
import logging
import os
from uuid import uuid4
import psycopg
from regchain import __version__
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from regchain.regulatory_api import router
from regchain.extraction_api import router as extraction_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("regchain")
app = FastAPI(title="RegChain", version=__version__, description="FCA sources, version history and grounded obligation candidates. All extraction results require review.")
app.include_router(router)
app.include_router(extraction_router)

class Health(BaseModel):
    status: str

@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info("request_id=%s method=%s status=%s", request_id, request.method, response.status_code)
    return response

@app.get("/health/live", response_model=Health)
def live():
    return Health(status="ok")

@app.get("/health/ready", response_model=Health)
def ready():
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=3) as conn:
            row = conn.execute("SELECT checksum FROM schema_migrations WHERE version = '008'").fetchone()
            if not row:
                raise ValueError("Schema not initialized")
    except (psycopg.Error, KeyError, ValueError):
        raise HTTPException(status_code=503, detail="Database not ready") from None
    return Health(status="ready")
