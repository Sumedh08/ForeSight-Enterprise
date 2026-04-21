from __future__ import annotations

import time

import structlog
from fastapi import Request

from safety.pii_mask import mask_text


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )


log = structlog.get_logger()


async def logging_middleware(request: Request, call_next):
    started = time.perf_counter()
    question_preview = ""
    if request.method == "POST" and request.headers.get("content-type", "").startswith("application/json"):
        try:
            payload = await request.json()
            question = payload.get("question", "") if isinstance(payload, dict) else ""
            question_preview = mask_text(str(question))
        except Exception:
            question_preview = ""
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    log.info(
        "request_complete",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
        question_preview=question_preview,
    )
    return response
