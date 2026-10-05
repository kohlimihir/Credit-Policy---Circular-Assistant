"""
Dense Retriever — Qdrant-backed vector search.

Manages collection creation, upsert, and top-K dense retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from config.settings import get_settings
from src.ingestion.chunking.base import Chunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Qdrant-backed dense vector retriever.

    Handles:
    - Collection creation with HNSW indexing
    - Batch upserting of chunk embeddings
    - Top-K cosine similarity search
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        self.embedding_dim = embedding_dim or settings.embedding_dim

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info(
            "DenseRetriever initialized — collection: %s, dim: %d",
            self.collection_name,
            self.embedding_dim,
        )

    def create_collection(self, recreate: bool = False) -> None:
        """Create or recreate the Qdrant collection."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            if recreate:
                self.client.delete_collection(self.collection_name)
                logger.info("Deleted existing collection: %s", self.collection_name)
            else:
                logger.info("Collection already exists: %s", self.collection_name)
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection: %s (dim=%d, cosine)",
            self.collection_name,
            self.embedding_dim,
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> None:
        """
        Upsert chunks with their embeddings into Qdrant.

        Args:
            chunks: List of Chunk objects.
            embeddings: Corresponding embedding vectors (N, dim).
            batch_size: Number of points per upsert batch.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        points: list[PointStruct] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "chunk_id": chunk.metadata.chunk_id,
                    "document_title": chunk.metadata.document_title,
                    "file_name": chunk.metadata.file_name,
                    "circular_number": chunk.metadata.circular_number,
                    "date": chunk.metadata.date,
                    "category": chunk.metadata.category,
                    "page_number": chunk.metadata.page_number,
                    "section_hierarchy": chunk.metadata.section_hierarchy,
                    "chunk_strategy": chunk.metadata.chunk_strategy,
                    "chunk_index": chunk.metadata.chunk_index,
                    "token_count": chunk.metadata.token_count,
                },
            )
            points.append(point)

        # Batch upsert
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(
            "Upserted %d points to collection '%s'",
            len(points),
            self.collection_name,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        strategy_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for the top-K most similar chunks.

        Args:
            query_embedding: Query vector (dim,).
            top_k: Number of results to return.
            strategy_filter: Optional filter by chunk_strategy.

        Returns:
            List of dicts with 'text', 'score', and metadata fields.
        """
        settings = get_settings()
        top_k = top_k or settings.dense_top_k

        query_filter = None
        if strategy_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="chunk_strategy",
                        match=MatchValue(value=strategy_filter),
                    )
                ]
            )

        # `search()` was removed in qdrant-client 1.13+; `query_points()` replaces it.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = response.points

        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id", ""),
                "document_title": hit.payload.get("document_title", ""),
                "file_name": hit.payload.get("file_name", ""),
                "circular_number": hit.payload.get("circular_number", ""),
                "page_number": hit.payload.get("page_number", 0),
                "section_hierarchy": hit.payload.get("section_hierarchy", []),
                "chunk_strategy": hit.payload.get("chunk_strategy", ""),
                "token_count": hit.payload.get("token_count", 0),
            }
            for hit in results
        ]

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
