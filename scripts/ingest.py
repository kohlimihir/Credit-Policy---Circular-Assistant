"""
Ingestion Pipeline Script — full end-to-end document ingestion.

Steps:
1. Parse all PDFs in the data directory
2. Extract metadata
3. Chunk using all three strategies
4. Embed chunks
5. Upsert to Qdrant
6. Build BM25 indices
7. Save all intermediate artifacts

Usage:
    python scripts/ingest.py                    # all strategies
    python scripts/ingest.py --strategy structural  # single strategy
    python scripts/ingest.py --recreate         # recreate collections
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.chunking import FixedChunker, RecursiveChunker, StructuralChunker
from src.ingestion.embedder import Embedder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion(
    strategy: str = "all",
    recreate: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        strategy: 'fixed_512', 'recursive_overlap', 'structural', or 'all'
        recreate: If True, recreate Qdrant collections

    Returns:
        Summary dict with counts and timings.
    """
    settings = get_settings()
    settings.ensure_dirs()
    t_total = time.perf_counter()

    # ── 1. Parse PDFs ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Parsing PDF documents")
    logger.info("=" * 60)

    parser = PDFParser()
    documents = parser.parse_directory(settings.pdf_dir)

    if not documents:
        logger.error("No PDF documents found in %s", settings.pdf_dir)
        logger.info("Run `python scripts/download_pdfs.py` first to download documents.")
        return {"error": "No documents found"}

    logger.info("Parsed %d documents successfully", len(documents))

    # ── 2. Extract Metadata ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting document metadata")
    logger.info("=" * 60)

    extractor = MetadataExtractor()
    metadata_list = []
    for doc in documents:
        meta = extractor.extract(doc)
        metadata_list.append(meta)

    # Save metadata
    meta_path = settings.processed_dir / "document_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            [m.to_dict() for m in metadata_list],
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved metadata to %s", meta_path)

    # ── 3. Chunk Documents ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Chunking documents")
    logger.info("=" * 60)

    chunkers = {
        "fixed_512": FixedChunker(settings.fixed_chunk_size),
        "recursive_overlap": RecursiveChunker(
            settings.recursive_chunk_size,
            settings.recursive_chunk_overlap,
        ),
        "structural": StructuralChunker(settings.structural_max_chunk_size),
    }

    if strategy != "all":
        if strategy not in chunkers:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(chunkers.keys())}")
        chunkers = {strategy: chunkers[strategy]}

    strategy_chunks = {}
    for name, chunker in chunkers.items():
        logger.info("--- Chunking with strategy: %s ---", name)
        chunks = []
        for doc, meta in zip(documents, metadata_list):
            doc_chunks = chunker.chunk(doc, meta)
            chunks.extend(doc_chunks)

        strategy_chunks[name] = chunks
        logger.info("  → %d chunks created", len(chunks))

        # Save chunks to JSON
        chunks_path = settings.processed_dir / f"chunks_{name}.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("  → Saved to %s", chunks_path)

    # ── 4. Embed and Index ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Embedding and indexing")
    logger.info("=" * 60)

    embedder = Embedder()

    for name, chunks in strategy_chunks.items():
        logger.info("--- Embedding strategy: %s (%d chunks) ---", name, len(chunks))

        t_embed = time.perf_counter()
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)
        embed_time = time.perf_counter() - t_embed
        logger.info("  → Embedded in %.1fs (%.1f chunks/sec)", embed_time, len(chunks) / embed_time)

        # Upsert to Qdrant
        collection_name = f"{settings.qdrant_collection}_{name}"
        dense = DenseRetriever(
            collection_name=collection_name,
            embedding_dim=embedder.dimension,
        )
        dense.create_collection(recreate=recreate)
        dense.upsert_chunks(chunks, embeddings)
        logger.info("  → Upserted to Qdrant collection: %s", collection_name)

        # Build BM25 index
        sparse = SparseRetriever()
        sparse.build_index(chunks)
        bm25_path = settings.processed_dir / f"bm25_index_{name}.json"
        sparse.save(bm25_path)
        logger.info("  → BM25 index saved to %s", bm25_path)

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total

    summary = {
        "documents_parsed": len(documents),
        "strategies": {
            name: {
                "chunks_created": len(chunks),
                "avg_tokens": sum(c.metadata.token_count for c in chunks) / len(chunks)
                if chunks else 0,
            }
            for name, chunks in strategy_chunks.items()
        },
        "total_time_seconds": round(total_time, 2),
    }

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Documents: %d", summary["documents_parsed"])
    for name, stats in summary["strategies"].items():
        logger.info(
            "  %s: %d chunks (avg %.0f tokens)",
            name,
            stats["chunks_created"],
            stats["avg_tokens"],
        )
    logger.info("Total time: %.1fs", total_time)

    # Save summary
    summary_path = settings.processed_dir / "ingestion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run document ingestion pipeline")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "fixed_512", "recursive_overlap", "structural"],
        help="Chunking strategy to use (default: all)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collections from scratch",
    )
    args = parser.parse_args()

    run_ingestion(strategy=args.strategy, recreate=args.recreate)
