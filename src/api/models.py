"""
Pydantic models — request/response schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for the /api/v1/query endpoint."""

    question: str = Field(
        ...,
        description="The underwriting / regulatory question to answer",
        min_length=5,
        max_length=2000,
        examples=["What is the minimum CRAR requirement for NBFC-ML?"],
    )
    strategy: str = Field(
        default="structural",
        description="Chunking strategy to use for retrieval",
        pattern="^(fixed_512|recursive_overlap|structural)$",
    )
    top_k: int = Field(
        default=5,
        description="Number of final results after reranking",
        ge=1,
        le=20,
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include retrieved context chunks in response",
    )


class IngestRequest(BaseModel):
    """Request body for the /api/v1/ingest endpoint."""

    pdf_directory: str = Field(
        default="",
        description="Directory containing PDFs to ingest (defaults to configured path)",
    )
    strategy: str = Field(
        default="all",
        description="Chunking strategy: 'fixed_512', 'recursive_overlap', 'structural', or 'all'",
    )
    recreate_collection: bool = Field(
        default=False,
        description="If true, recreate the Qdrant collection from scratch",
    )


# ── Response Models ──────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single citation reference."""

    document_title: str
    section: str = ""
    page: int | None = None
    verified: bool = True
    matched_chunk_id: str = ""


class ContextChunk(BaseModel):
    """A retrieved context chunk included in the response."""

    chunk_id: str
    text: str
    document_title: str
    section_hierarchy: list[str] = []
    page_number: int = 0
    score: float = 0.0
    rerank_score: float = 0.0


class AbstentionInfo(BaseModel):
    """Information about the abstention decision."""

    abstained: bool
    reason: str = ""
    max_score: float = 0.0
    threshold: float = 0.0


class TimingInfo(BaseModel):
    """Latency breakdown for the query."""

    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0


class TokenUsage(BaseModel):
    """Token usage for the LLM generation."""

    prompt: int = 0
    completion: int = 0
    total: int = 0


class QueryResponse(BaseModel):
    """Response body for the /api/v1/query endpoint."""

    answer: str
    citations: list[Citation] = []
    abstention: AbstentionInfo
    context_chunks: list[ContextChunk] = []
    timing: TimingInfo
    token_usage: TokenUsage
    strategy: str = ""
    verified_citations: int = 0
    unverified_citations: int = 0


class HealthResponse(BaseModel):
    """Response body for the /api/v1/health endpoint."""

    status: str = "healthy"
    version: str = "1.0.0"
    qdrant_connected: bool = False
    documents_indexed: int = 0
    embedding_model: str = ""
    reranker_model: str = ""


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    file_name: str
    title: str
    circular_number: str = ""
    date: str = ""
    category: str = ""
    total_pages: int = 0
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    """Response body for the /api/v1/documents endpoint."""

    documents: list[DocumentInfo] = []
    total: int = 0


class IngestResponse(BaseModel):
    """Response body for the /api/v1/ingest endpoint."""

    status: str
    documents_processed: int = 0
    chunks_created: int = 0
    strategy: str = ""
    duration_seconds: float = 0.0


class MetricsResponse(BaseModel):
    """Response body for the /api/v1/metrics endpoint."""

    strategies: list[dict] = []
    last_evaluated: str = ""
