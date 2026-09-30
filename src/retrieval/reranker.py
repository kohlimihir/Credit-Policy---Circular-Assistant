"""
Cross-Encoder Reranker — reranks hybrid retrieval candidates using
a cross-encoder model for fine-grained relevance scoring.

Cross-encoders process (query, passage) pairs jointly, producing much
more accurate relevance scores than bi-encoder dot products, at the
cost of being too slow for first-stage retrieval.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sentence_transformers import CrossEncoder

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for second-stage retrieval.

    Takes the top-N candidates from hybrid search and produces
    fine-grained relevance scores for final ranking.

    The max reranked score is also used for the abstention decision.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model

        logger.info("Loading cross-encoder reranker: %s", self.model_name)
        self.model = CrossEncoder(self.model_name)
        logger.info("Cross-encoder reranker loaded")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query: The user's question.
            candidates: List of candidate dicts from hybrid retrieval.
                       Each must have a 'text' key.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of result dicts with 'rerank_score' added.
        """
        settings = get_settings()
        top_k = top_k or settings.rerank_top_k

        if not candidates:
            return []

        t_start = time.perf_counter()

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_candidates: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            result["rerank_score"] = float(score)
            result["pre_rerank_score"] = result.get("score", 0)
            result["score"] = float(score)  # update score to reranked value
            scored_candidates.append(result)

        # Sort by rerank score descending
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        rerank_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Reranked %d candidates → top %d in %.0fms "
            "(top_score=%.4f, bottom_score=%.4f)",
            len(candidates),
            top_k,
            rerank_ms,
            scored_candidates[0]["rerank_score"] if scored_candidates else 0,
            scored_candidates[min(top_k, len(scored_candidates)) - 1]["rerank_score"]
            if scored_candidates
            else 0,
        )

        # Return top-K
        top_results = scored_candidates[:top_k]

        # Attach timing
        for r in top_results:
            r.setdefault("_timing", {})
            r["_timing"]["rerank_ms"] = round(rerank_ms, 1)

        return top_results

    def get_max_score(self, reranked_results: list[dict[str, Any]]) -> float:
        """Return the maximum rerank score from results (for abstention check)."""
        if not reranked_results:
            return 0.0
        return max(r.get("rerank_score", 0.0) for r in reranked_results)
