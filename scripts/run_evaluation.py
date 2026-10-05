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

    # Retrievers are built once per chunking strategy and reused. Building them
    # per question would reload and re-tokenize the BM25 index (several MB of
    # JSON) on every call, which dominated runtime in the original version.
    retriever_cache: dict[str, HybridRetriever] = {}

    def get_retriever(base_strategy: str) -> HybridRetriever:
        if base_strategy not in retriever_cache:
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
                raise FileNotFoundError(
                    f"BM25 index missing: {bm25_path}. "
                    "Run `python scripts/ingest.py --strategy all` first."
                )

            retriever_cache[base_strategy] = HybridRetriever(embedder, dense, sparse)
            logger.info("Initialized retrievers for strategy '%s'", base_strategy)

        return retriever_cache[base_strategy]

    def query_fn(question: str, strategy: str) -> dict[str, Any]:
        """Execute the full RAG pipeline for a single question."""
        t_start = time.perf_counter()

        # Reranking is applied to EVERY strategy, so the chunking comparison is
        # apples-to-apples. The `_norerank` suffix is the deliberate ablation
        # used to quantify how much the reranker itself contributes.
        use_rerank = "norerank" not in strategy
        base_strategy = strategy.replace("_norerank", "")

        hybrid = get_retriever(base_strategy)

        # Stage 1: hybrid retrieval (dense + BM25, fused with RRF)
        candidates = hybrid.search(query=question, top_k=50)

        # Stage 2: rerank to the final context window
        if use_rerank:
            final = reranker.rerank(question, candidates, top_k=settings.rerank_top_k)
        else:
            final = candidates[: settings.rerank_top_k]

        # Stage 3: abstention. The threshold is calibrated against cross-encoder
        # scores, so it is only meaningful when reranking ran. For the ablation
        # the check is skipped rather than applied to an incomparable score.
        if use_rerank:
            abstention_result = abstention.should_abstain(final)
            abstained = abstention_result["abstain"]
        else:
            abstained = False

        answer = ""
        if not abstained:
            try:
                gen_result = generator.generate(question, final)
                answer = gen_result.get("answer", "")
            except Exception as e:
                logger.error("Generation failed: %s", str(e))
                answer = f"Generation error: {str(e)}"

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "retrieved_ids": [c.get("chunk_id", "") for c in final],
            # Source document of each chunk in the final context window, in rank
            # order. Retrieval metrics are scored against this.
            "retrieved_documents": [
                c.get("file_name") or c.get("document_title", "") for c in final
            ],
            "answer": answer,
            "context_chunks": final,
            "abstained": abstained,
            "latency_ms": latency_ms,
        }

    return query_fn


def load_document_aliases() -> dict[str, str]:
    """
    Map gold-set document names onto the files actually on disk.

    RBI consolidated several NBFC directions in November 2025, so two gold
    targets can resolve to one source PDF (the NBFC interest-rate rules now sit
    inside the Responsible Business Conduct directions). The scraper records
    that in data/raw_pdfs/manifest.json as `status: alias`, and this reads it
    back so those questions are scored against the file that really holds the
    content instead of scoring zero against a filename that was never saved.
    """
    settings = get_settings()
    manifest_path = settings.pdf_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    aliases = {
        entry["file_name"]: entry["alias_of"]
        for entry in manifest
        if entry.get("status") == "alias" and entry.get("alias_of")
    }
    if aliases:
        logger.info("Applying %d document alias(es) from manifest: %s", len(aliases), aliases)
    return aliases


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

    # Remap gold document names onto the files on disk before scoring
    aliases = load_document_aliases()
    if aliases:
        for entry in runner.evaluator.gold_set:
            entry["relevant_documents"] = [
                aliases.get(d, d) for d in entry.get("relevant_documents", [])
            ]

    # Warn about any gold document that is not in the corpus: those questions
    # can never score, and silently reporting a lower recall would be misleading.
    on_disk = {p.name for p in settings.pdf_dir.glob("*.pdf")}
    referenced = {
        d for entry in runner.evaluator.gold_set for d in entry.get("relevant_documents", [])
    }
    missing = referenced - on_disk
    if missing:
        affected = sum(
            1
            for entry in runner.evaluator.gold_set
            if missing & set(entry.get("relevant_documents", []))
        )
        logger.warning(
            "%d gold document(s) absent from the corpus, affecting %d question(s): %s",
            len(missing),
            affected,
            sorted(missing),
        )

    if strategy:
        # Single strategy
        metrics = runner.run_strategy_benchmark(strategy, query_fn)
        logger.info("Results for %s:", strategy)
        logger.info(json.dumps(metrics, indent=2, default=str))
    else:
        # All strategies
        all_metrics = runner.run_all_strategies(query_fn, include_ablation=True)
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
        choices=["fixed_512", "recursive_overlap", "structural", "structural_norerank"],
        help="Run evaluation for a single strategy only",
    )
    args = parser.parse_args()

    run_evaluation(gold_set_path=args.gold_set, strategy=args.strategy)
