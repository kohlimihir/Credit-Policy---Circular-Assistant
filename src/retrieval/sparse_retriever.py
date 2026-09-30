"""
Sparse Retriever — BM25-based sparse retrieval.

Builds an in-memory BM25 index over chunk texts for lexical matching.
Serializable to disk for persistence across restarts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import get_settings
from src.ingestion.chunking.base import Chunk

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    BM25-based sparse retriever for lexical matching.

    Complements dense retrieval by catching exact keyword matches
    (e.g., "NBFC-ML", "Section 45-IA", "DoR.FIN.REC.21")
    that embedding models may miss.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[Chunk]) -> None:
        """
        Build the BM25 index from a list of chunks.

        Args:
            chunks: List of Chunk objects to index.
        """
        self._chunks = [
            {
                "text": chunk.text,
                "chunk_id": chunk.metadata.chunk_id,
                "document_title": chunk.metadata.document_title,
                "file_name": chunk.metadata.file_name,
                "circular_number": chunk.metadata.circular_number,
                "page_number": chunk.metadata.page_number,
                "section_hierarchy": chunk.metadata.section_hierarchy,
                "chunk_strategy": chunk.metadata.chunk_strategy,
                "token_count": chunk.metadata.token_count,
            }
            for chunk in chunks
        ]

        # Tokenize corpus for BM25
        self._tokenized_corpus = [
            self._tokenize(chunk.text) for chunk in chunks
        ]

        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(chunks))

    def search(
        self,
        query: str,
        top_k: int | None = None,
        strategy_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the BM25 index.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            strategy_filter: Optional filter by chunk_strategy.

        Returns:
            List of dicts with 'text', 'score', and metadata.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        settings = get_settings()
        top_k = top_k or settings.sparse_top_k

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-K indices
        scored_indices = list(enumerate(scores))

        # Apply strategy filter if specified
        if strategy_filter:
            scored_indices = [
                (i, s)
                for i, s in scored_indices
                if self._chunks[i].get("chunk_strategy") == strategy_filter
            ]

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]

        results: list[dict[str, Any]] = []
        for idx, score in top_indices:
            if score <= 0:
                continue
            result = dict(self._chunks[idx])
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, path: str | Path) -> None:
        """Save the index data to disk (JSON + tokenized corpus)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info("BM25 index saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load the index data from disk."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index loaded from %s (%d documents)", path, len(self._chunks))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Simple tokenization optimized for regulatory documents.

        Preserves:
        - Alphanumeric tokens (lowercased)
        - Regulatory identifiers (e.g., "nbfc-ml", "section-45")
        - Numbers and dates
        """
        # Lowercase and split on non-alphanumeric (keeping hyphens in compound terms)
        text = text.lower()
        # Split but keep hyphenated terms together
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
        # Remove very short tokens (< 2 chars) unless they're numbers
        tokens = [t for t in tokens if len(t) >= 2 or t.isdigit()]
        return tokens
