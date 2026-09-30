"""
Structured Logging Middleware — logs every query with full retrieval
and generation metadata for observability and debugging.
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON-formatted structured logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with timing and metadata."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Attach request ID to request state
        request.state.request_id = request_id

        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 1),
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Duration-Ms"] = str(round(duration_ms, 1))

        return response


def log_query(
    request_id: str,
    question: str,
    strategy: str,
    retrieved_chunk_ids: list[str],
    rerank_scores: list[float],
    abstained: bool,
    token_count: dict[str, int],
    latency_breakdown: dict[str, float],
    answer_length: int,
    citation_count: int,
) -> None:
    """
    Log a complete query event with all retrieval and generation metadata.

    This is called after each query to create a structured log entry
    for debugging, monitoring, and analysis.
    """
    logger.info(
        "query_processed",
        request_id=request_id,
        question=question[:200],
        strategy=strategy,
        retrieved_chunks=len(retrieved_chunk_ids),
        chunk_ids=retrieved_chunk_ids[:10],  # top 10
        top_rerank_score=round(max(rerank_scores), 4) if rerank_scores else 0,
        abstained=abstained,
        prompt_tokens=token_count.get("prompt", 0),
        completion_tokens=token_count.get("completion", 0),
        total_tokens=token_count.get("total", 0),
        retrieval_ms=latency_breakdown.get("retrieval_ms", 0),
        rerank_ms=latency_breakdown.get("rerank_ms", 0),
        generation_ms=latency_breakdown.get("generation_ms", 0),
        total_ms=latency_breakdown.get("total_ms", 0),
        answer_length=answer_length,
        citation_count=citation_count,
    )
