"""
Evaluation Runner — runs the benchmark across all chunking strategies
and produces the comparison table.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --strategy structural
    python scripts/run_evaluation.py --gold-set path/to/gold_set.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from src.ingestion.embedder import Embedder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generator import RAGGenerator
from src.generation.abstention import AbstentionChecker
from src.generation.citation_verifier import CitationVerifier
from src.evaluation.benchmark import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_query_fn(
    embedder: Embedder,
    reranker: CrossEncoderReranker,
    generator: RAGGenerator,
    abstention: AbstentionChecker,
    verifier: CitationVerifier,
):
    """Build the query function that the benchmark runner will call."""
    settings = get_settings()

    def query_fn(question: str, strategy: str) -> dict[str, Any]:
        """Execute the full RAG pipeline for a single question."""
        t_start = time.perf_counter()

        # Determine if this is a reranked run
        use_rerank = "reranked" in strategy
        base_strategy = strategy.replace("_reranked", "")

        # Set up retriever for this strategy
        collection_name = f"{settings.qdrant_collection}_{base_strategy}"
        dense = DenseRetriever(
            collection_name=collection_name,
            embedding_dim=embedder.dimension,
        )

        sparse = SparseRetriever()
        bm25_path = settings.processed_dir / f"bm25_index_{base_strategy}.json"
        if bm25_path.exists():
            sparse.load(bm25_path)
        else:
            logger.warning("BM25 index not found: %s", bm25_path)

        hybrid = HybridRetriever(embedder, dense, sparse)

        # Hybrid retrieval
        candidates = hybrid.search(
            query=question,
            top_k=50,
        )

        retrieved_ids = [c.get("chunk_id", "") for c in candidates]

        # Reranking (always for reranked strategy, optionally for others)
        if use_rerank:
            reranked = reranker.rerank(question, candidates, top_k=settings.rerank_top_k)
        else:
            reranked = candidates[:settings.rerank_top_k]

        # Abstention check
        abstention_result = abstention.should_abstain(reranked)
        abstained = abstention_result["abstain"]

        answer = ""
        context_chunks = reranked

        if not abstained:
            try:
                gen_result = generator.generate(question, reranked)
                answer = gen_result.get("answer", "")
            except Exception as e:
                logger.error("Generation failed: %s", str(e))
                answer = f"Generation error: {str(e)}"

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "retrieved_ids": retrieved_ids[:10],
            "answer": answer,
            "context_chunks": context_chunks,
            "abstained": abstained,
            "latency_ms": latency_ms,
        }

    return query_fn


def run_evaluation(
    gold_set_path: str | None = None,
    strategy: str | None = None,
) -> None:
    """Run the full evaluation benchmark."""
    settings = get_settings()

    # Default gold set path
    if gold_set_path is None:
        gold_set_path = str(Path(__file__).parent.parent / "src" / "evaluation" / "gold_set.json")

    if not Path(gold_set_path).exists():
        logger.error("Gold set not found: %s", gold_set_path)
        logger.info("Create the gold set first at src/evaluation/gold_set.json")
        return

    logger.info("=" * 60)
    logger.info("EVALUATION BENCHMARK")
    logger.info("=" * 60)

    # Initialize components
    logger.info("Loading models...")
    embedder = Embedder()
    reranker = CrossEncoderReranker()
    generator = RAGGenerator()
    abstention_checker = AbstentionChecker()
    verifier = CitationVerifier()

    # Build query function
    query_fn = build_query_fn(embedder, reranker, generator, abstention_checker, verifier)

    # Run benchmark
    runner = BenchmarkRunner(
        gold_set_path=gold_set_path,
        output_dir=settings.processed_dir / "benchmarks",
    )

    if strategy:
        # Single strategy
        metrics = runner.run_strategy_benchmark(strategy, query_fn)
        logger.info("Results for %s:", strategy)
        logger.info(json.dumps(metrics, indent=2, default=str))
    else:
        # All strategies
        all_metrics = runner.run_all_strategies(query_fn, include_reranked=True)
        logger.info("\nAll strategy results:")
        for m in all_metrics:
            logger.info("  %s: Recall@5=%.4f, MRR=%.4f",
                       m["strategy"],
                       m["retrieval"]["recall_at_5"],
                       m["retrieval"]["mrr"])

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info("Results saved to: %s", settings.processed_dir / "benchmarks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument(
        "--gold-set",
        default=None,
        help="Path to gold evaluation set JSON",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["fixed_512", "recursive_overlap", "structural", "structural_reranked"],
        help="Run evaluation for a single strategy only",
    )
    args = parser.parse_args()

    run_evaluation(gold_set_path=args.gold_set, strategy=args.strategy)
