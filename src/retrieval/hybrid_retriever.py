"""
Hybrid Retriever — fuses BM25 sparse and dense vector search results
using Reciprocal Rank Fusion (RRF).

RRF_score(d) = Σ 1/(k + rank_i(d))

This produces a single ranked list from multiple rankers without
requiring score normalization.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from config.settings import get_settings
from src.ingestion.embedder import Embedder
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining dense + sparse search with RRF fusion.

    Pipeline:
    1. Run dense vector search → top K candidates
    2. Run BM25 sparse search → top K candidates
    3. Fuse with Reciprocal Rank Fusion
    4. Return top K fused results
    """

    def __init__(
        self,
        embedder: Embedder,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int | None = None,
    ) -> None:
        self.embedder = embedder
        self.dense = dense_retriever
        self.sparse = sparse_retriever

        settings = get_settings()
        self.rrf_k = rrf_k or settings.rrf_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        strategy_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute hybrid search: dense + sparse → RRF fusion.

        Args:
            query: User's search query.
            top_k: Number of final fused results.
            dense_top_k: Number of dense retrieval candidates.
            sparse_top_k: Number of sparse retrieval candidates.
            strategy_filter: Optional filter by chunking strategy.

        Returns:
            Fused and ranked list of result dicts.
        """
        settings = get_settings()
        top_k = top_k or settings.rerank_top_k
        dense_top_k = dense_top_k or settings.dense_top_k
        sparse_top_k = sparse_top_k or settings.sparse_top_k

        t_start = time.perf_counter()

        # 1. Dense search
        t_embed = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t_embed) * 1000

        t_dense = time.perf_counter()
        dense_results = self.dense.search(
            query_embedding,
            top_k=dense_top_k,
            strategy_filter=strategy_filter,
        )
        dense_ms = (time.perf_counter() - t_dense) * 1000

        # 2. Sparse search
        t_sparse = time.perf_counter()
        sparse_results = self.sparse.search(
            query,
            top_k=sparse_top_k,
            strategy_filter=strategy_filter,
        )
        sparse_ms = (time.perf_counter() - t_sparse) * 1000

        # 3. RRF fusion
        t_fuse = time.perf_counter()
        fused_results = self._reciprocal_rank_fusion(
            [dense_results, sparse_results],
            top_k=max(top_k, 50),  # keep at least 50 for reranker
        )
        fuse_ms = (time.perf_counter() - t_fuse) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Hybrid search: query='%s' — dense=%d results (%.0fms), "
            "sparse=%d results (%.0fms), fused=%d results (%.0fms), "
            "total=%.0fms",
            query[:80],
            len(dense_results),
            dense_ms,
            len(sparse_results),
            sparse_ms,
            len(fused_results),
            fuse_ms,
            total_ms,
        )

        # Attach timing metadata to results
        for r in fused_results:
            r["_timing"] = {
                "embed_ms": round(embed_ms, 1),
                "dense_ms": round(dense_ms, 1),
                "sparse_ms": round(sparse_ms, 1),
                "fuse_ms": round(fuse_ms, 1),
                "total_ms": round(total_ms, 1),
            }

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[dict[str, Any]]],
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fuse multiple ranked result lists using RRF.

        RRF_score(d) = Σ 1/(k + rank_i(d))

        Uses chunk_id as the unique key for deduplication.
        """
        # Map chunk_id → { accumulated RRF score, best result dict }
        rrf_scores: dict[str, float] = {}
        best_result: dict[str, dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                chunk_id = result.get("chunk_id", "")
                if not chunk_id:
                    continue

                rrf_score = 1.0 / (self.rrf_k + rank + 1)  # rank is 0-indexed
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

                # Keep the result dict with the best individual score
                if chunk_id not in best_result or result.get("score", 0) > best_result[chunk_id].get("score", 0):
                    best_result[chunk_id] = result

        # Sort by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused: list[dict[str, Any]] = []
        for chunk_id in sorted_ids[:top_k]:
            result = dict(best_result[chunk_id])
            result["rrf_score"] = rrf_scores[chunk_id]
            result["original_score"] = result.get("score", 0)
            result["score"] = rrf_scores[chunk_id]  # replace with fused score
            fused.append(result)

        return fused
