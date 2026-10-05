"""
FastAPI Application — serves the Credit Policy & Circular Assistant.

Endpoints:
  POST /api/v1/query      — Ask a question with citations
  GET  /api/v1/health      — Health check
  GET  /api/v1/documents   — List indexed documents
  POST /api/v1/ingest      — Trigger document ingestion
  GET  /api/v1/metrics     — Evaluation metrics
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from src.api.middleware import (
    RequestLoggingMiddleware,
    configure_logging,
    log_query,
)
from src.api.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    AbstentionInfo,
    TimingInfo,
    TokenUsage,
    ContextChunk,
    Citation,
)

logger = logging.getLogger(__name__)

# ── Global pipeline components (initialized at startup) ──────────────────
_pipeline: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline components on startup, cleanup on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("Initializing pipeline components...")

    try:
        from src.ingestion.embedder import Embedder
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker
        from src.generation.generator import RAGGenerator
        from src.generation.abstention import AbstentionChecker
        from src.generation.citation_verifier import CitationVerifier

        embedder = Embedder()
        dense = DenseRetriever()
        sparse = SparseRetriever()

        # Try to load saved BM25 index
        bm25_path = settings.processed_dir / "bm25_index.json"
        if bm25_path.exists():
            sparse.load(bm25_path)
            logger.info("Loaded BM25 index from %s", bm25_path)
        else:
            logger.warning("No BM25 index found at %s — sparse search disabled until ingestion", bm25_path)

        hybrid = HybridRetriever(embedder, dense, sparse)
        reranker = CrossEncoderReranker()
        generator = RAGGenerator()
        abstention = AbstentionChecker()
        verifier = CitationVerifier()

        _pipeline.update({
            "embedder": embedder,
            "dense": dense,
            "sparse": sparse,
            "hybrid": hybrid,
            "reranker": reranker,
            "generator": generator,
            "abstention": abstention,
            "verifier": verifier,
            "ready": True,
        })
        logger.info("Pipeline initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize pipeline: %s", str(e))
        _pipeline["ready"] = False
        _pipeline["error"] = str(e)

    yield

    # Cleanup
    _pipeline.clear()
    logger.info("Pipeline shutdown complete")


# ── App Factory ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Policy & Circular Assistant",
    description=(
        "RAG-powered question-answering system over RBI master directions, "
        "NBFC credit policy documents, and product circulars. "
        "Answers with exact clause-level citations and refuses when context "
        "does not support an answer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structured logging
app.add_middleware(RequestLoggingMiddleware)


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a regulatory / underwriting question with citations.

    Full pipeline: hybrid retrieval → reranking → abstention check →
    LLM generation → citation verification.
    """
    if not _pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not initialized")

    t_total_start = time.perf_counter()
    question = request.question
    strategy = request.strategy

    # ── 1. Hybrid Retrieval ──────────────────────────────────────────
    t_retrieval = time.perf_counter()
    hybrid_results = _pipeline["hybrid"].search(
        query=question,
        top_k=50,
        strategy_filter=strategy,
    )
    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

    # ── 2. Reranking ─────────────────────────────────────────────────
    t_rerank = time.perf_counter()
    reranked = _pipeline["reranker"].rerank(
        query=question,
        candidates=hybrid_results,
        top_k=request.top_k,
    )
    rerank_ms = (time.perf_counter() - t_rerank) * 1000

    # ── 3. Abstention Check ──────────────────────────────────────────
    abstention_result = _pipeline["abstention"].should_abstain(reranked)

    if abstention_result["abstain"]:
        abstention_response = _pipeline["abstention"].get_abstention_response()
        total_ms = (time.perf_counter() - t_total_start) * 1000

        # Log the query
        log_query(
            request_id="N/A",
            question=question,
            strategy=strategy,
            retrieved_chunk_ids=[r.get("chunk_id", "") for r in reranked],
            rerank_scores=[r.get("rerank_score", 0) for r in reranked],
            abstained=True,
            token_count={"prompt": 0, "completion": 0, "total": 0},
            latency_breakdown={
                "retrieval_ms": retrieval_ms,
                "rerank_ms": rerank_ms,
                "generation_ms": 0,
                "total_ms": total_ms,
            },
            answer_length=len(abstention_response["answer"]),
            citation_count=0,
        )

        return QueryResponse(
            answer=abstention_response["answer"],
            citations=[],
            abstention=AbstentionInfo(
                abstained=True,
                reason=abstention_result["reason"],
                max_score=abstention_result["max_score"],
                threshold=abstention_result["threshold"],
            ),
            context_chunks=[
                ContextChunk(
                    chunk_id=r.get("chunk_id", ""),
                    text=r.get("text", "")[:500],
                    document_title=r.get("document_title", ""),
                    section_hierarchy=r.get("section_hierarchy", []),
                    page_number=r.get("page_number", 0),
                    score=r.get("original_score", 0),
                    rerank_score=r.get("rerank_score", 0),
                )
                for r in reranked
            ] if request.include_context else [],
            timing=TimingInfo(
                total_ms=total_ms,
                retrieval_ms=retrieval_ms,
                rerank_ms=rerank_ms,
                generation_ms=0,
            ),
            token_usage=TokenUsage(),
            strategy=strategy,
        )

    # ── 4. Generation ────────────────────────────────────────────────
    t_gen = time.perf_counter()
    try:
        gen_result = _pipeline["generator"].generate(
            question=question,
            context_chunks=reranked,
        )
    except Exception as e:
        # Upstream LLM failures (bad API key, quota, network) are not server bugs
        logger.exception("LLM generation failed")
        raise HTTPException(
            502,
            f"LLM generation failed ({type(e).__name__}). "
            "Check OPENAI_API_KEY / GOOGLE_API_KEY and LLM_PROVIDER in .env.",
        )
    generation_ms = (time.perf_counter() - t_gen) * 1000

    # ── 5. Citation Verification ─────────────────────────────────────
    verification = _pipeline["verifier"].verify_and_clean(
        answer=gen_result["answer"],
        retrieved_chunks=reranked,
    )

    total_ms = (time.perf_counter() - t_total_start) * 1000

    # ── 6. Structured Logging ────────────────────────────────────────
    log_query(
        request_id="N/A",
        question=question,
        strategy=strategy,
        retrieved_chunk_ids=[r.get("chunk_id", "") for r in reranked],
        rerank_scores=[r.get("rerank_score", 0) for r in reranked],
        abstained=False,
        token_count=gen_result.get("token_usage", {}),
        latency_breakdown={
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
        answer_length=len(verification["cleaned_answer"]),
        citation_count=verification["verified_count"],
    )

    # ── 7. Build Response ────────────────────────────────────────────
    return QueryResponse(
        answer=verification["cleaned_answer"],
        citations=[
            Citation(
                document_title=c["document_title"],
                section=c.get("section", ""),
                page=c.get("page"),
                verified=c["verified"],
                matched_chunk_id=c.get("matched_chunk_id", ""),
            )
            for c in verification["citations"]
        ],
        abstention=AbstentionInfo(
            abstained=False,
            reason=abstention_result["reason"],
            max_score=abstention_result["max_score"],
            threshold=abstention_result["threshold"],
        ),
        context_chunks=[
            ContextChunk(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", "")[:500] if request.include_context else "",
                document_title=r.get("document_title", ""),
                section_hierarchy=r.get("section_hierarchy", []),
                page_number=r.get("page_number", 0),
                score=r.get("original_score", 0),
                rerank_score=r.get("rerank_score", 0),
            )
            for r in reranked
        ] if request.include_context else [],
        timing=TimingInfo(
            total_ms=round(total_ms, 1),
            retrieval_ms=round(retrieval_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            generation_ms=round(generation_ms, 1),
        ),
        token_usage=TokenUsage(**gen_result.get("token_usage", {})),
        strategy=strategy,
        verified_citations=verification["verified_count"],
        unverified_citations=verification["unverified_count"],
    )


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()
    qdrant_connected = False
    docs_indexed = 0

    if _pipeline.get("ready"):
        try:
            docs_indexed = _pipeline["dense"].count()
            qdrant_connected = True
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if _pipeline.get("ready") else "degraded",
        qdrant_connected=qdrant_connected,
        documents_indexed=docs_indexed,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )


@app.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all indexed documents."""
    # This would query Qdrant for unique document titles
    # For now, return from processed metadata
    settings = get_settings()
    meta_path = settings.processed_dir / "document_metadata.json"

    if not meta_path.exists():
        return DocumentListResponse(documents=[], total=0)

    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    from src.api.models import DocumentInfo
    doc_infos = [
        DocumentInfo(
            file_name=d.get("file_name", ""),
            title=d.get("title", ""),
            circular_number=d.get("circular_number", ""),
            date=d.get("date", ""),
            category=d.get("category", ""),
            total_pages=d.get("total_pages", 0),
        )
        for d in docs
    ]

    return DocumentListResponse(documents=doc_infos, total=len(doc_infos))


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Trigger document ingestion."""
    if not _pipeline.get("ready"):
        raise HTTPException(503, "Pipeline not initialized")

    settings = get_settings()
    pdf_dir = request.pdf_directory or str(settings.pdf_dir)

    t_start = time.perf_counter()

    try:
        from src.ingestion.pdf_parser import PDFParser
        from src.ingestion.metadata_extractor import MetadataExtractor
        from src.ingestion.chunking import FixedChunker, RecursiveChunker, StructuralChunker

        parser = PDFParser()
        extractor = MetadataExtractor()

        # Parse all PDFs
        documents = parser.parse_directory(pdf_dir)

        if not documents:
            return IngestResponse(
                status="no_documents_found",
                documents_processed=0,
            )

        # Extract metadata
        all_metadata = []
        for doc in documents:
            meta = extractor.extract(doc)
            all_metadata.append(meta.to_dict())

        # Save metadata
        import json
        settings.ensure_dirs()
        meta_path = settings.processed_dir / "document_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        # Select chunking strategy
        strategies = {
            "fixed_512": FixedChunker(settings.fixed_chunk_size),
            "recursive_overlap": RecursiveChunker(
                settings.recursive_chunk_size,
                settings.recursive_chunk_overlap,
            ),
            "structural": StructuralChunker(settings.structural_max_chunk_size),
        }

        total_chunks = 0
        if request.strategy == "all":
            for name, chunker in strategies.items():
                chunks = []
                for doc, meta_dict in zip(documents, all_metadata):
                    from src.ingestion.metadata_extractor import DocumentMetadata
                    meta_obj = extractor.extract(doc)
                    doc_chunks = chunker.chunk(doc, meta_obj)
                    chunks.extend(doc_chunks)

                # Embed and upsert
                texts = [c.text for c in chunks]
                embeddings = _pipeline["embedder"].embed_documents(texts)

                collection_name = f"{settings.qdrant_collection}_{name}"
                dense = DenseRetriever(
                    collection_name=collection_name,
                    embedding_dim=_pipeline["embedder"].dimension,
                )
                from src.retrieval.dense_retriever import DenseRetriever
                dense.create_collection(recreate=request.recreate_collection)
                dense.upsert_chunks(chunks, embeddings)

                # Build BM25 index
                _pipeline["sparse"].build_index(chunks)
                _pipeline["sparse"].save(
                    settings.processed_dir / f"bm25_index_{name}.json"
                )

                total_chunks += len(chunks)
        else:
            chunker = strategies.get(request.strategy)
            if not chunker:
                raise HTTPException(400, f"Unknown strategy: {request.strategy}")

            chunks = []
            for doc in documents:
                meta_obj = extractor.extract(doc)
                doc_chunks = chunker.chunk(doc, meta_obj)
                chunks.extend(doc_chunks)

            texts = [c.text for c in chunks]
            embeddings = _pipeline["embedder"].embed_documents(texts)

            _pipeline["dense"].create_collection(recreate=request.recreate_collection)
            _pipeline["dense"].upsert_chunks(chunks, embeddings)

            _pipeline["sparse"].build_index(chunks)
            _pipeline["sparse"].save(settings.processed_dir / "bm25_index.json")

            total_chunks = len(chunks)

        duration = time.perf_counter() - t_start

        return IngestResponse(
            status="success",
            documents_processed=len(documents),
            chunks_created=total_chunks,
            strategy=request.strategy,
            duration_seconds=round(duration, 2),
        )

    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Return latest evaluation metrics."""
    settings = get_settings()
    comparison_path = settings.processed_dir / "benchmarks" / "strategy_comparison.json"

    if not comparison_path.exists():
        return MetricsResponse(strategies=[], last_evaluated="never")

    import json
    with open(comparison_path, "r", encoding="utf-8") as f:
        strategies = json.load(f)

    return MetricsResponse(
        strategies=strategies,
        last_evaluated=str(comparison_path.stat().st_mtime),
    )


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
