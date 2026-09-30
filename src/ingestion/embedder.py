"""
Embedder — wraps sentence-transformers for generating dense embeddings.

Uses BAAI/bge-base-en-v1.5 by default (768-dim).  Supports batch encoding
with progress tracking for large document sets.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Dense embedding generator using sentence-transformers.

    For BGE models, queries should be prefixed with "Represent this sentence: "
    to match the training distribution.
    """

    # BGE models use an instruction prefix for queries
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._is_bge = "bge" in self.model_name.lower()

        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded — dimension: %d, is_bge: %s",
            self.dimension,
            self._is_bge,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a batch of document texts (no query prefix).

        Args:
            texts: Sequence of document text strings.
            batch_size: Batch size for encoding.
            show_progress: Show tqdm progress bar.

        Returns:
            numpy array of shape (len(texts), dimension).
        """
        logger.info("Embedding %d documents (batch_size=%d)", len(texts), batch_size)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine sim
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string (with BGE prefix if applicable).

        Args:
            query: The user's search query.

        Returns:
            numpy array of shape (dimension,).
        """
        if self._is_bge:
            query = f"{self.BGE_QUERY_PREFIX}{query}"

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return np.array(embedding)

    def embed_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a batch of queries (with BGE prefix if applicable).

        Args:
            queries: Sequence of query strings.
            batch_size: Batch size for encoding.

        Returns:
            numpy array of shape (len(queries), dimension).
        """
        if self._is_bge:
            queries = [f"{self.BGE_QUERY_PREFIX}{q}" for q in queries]
        else:
            queries = list(queries)

        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return np.array(embeddings)
