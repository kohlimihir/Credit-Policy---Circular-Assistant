"""
Base Chunker — abstract interface and shared data structures for all chunking strategies.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.ingestion.pdf_parser import ParsedDocument
from src.ingestion.metadata_extractor import DocumentMetadata


@dataclass
class ChunkMetadata:
    """Metadata attached to every chunk, used for retrieval and citations."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_title: str = ""
    file_name: str = ""
    file_hash: str = ""
    circular_number: str = ""
    date: str = ""
    category: str = ""
    page_number: int = 0
    page_numbers: list[int] = field(default_factory=list)  # if chunk spans pages
    section_hierarchy: list[str] = field(default_factory=list)
    chunk_strategy: str = ""
    chunk_index: int = 0
    token_count: int = 0
    char_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "circular_number": self.circular_number,
            "date": self.date,
            "category": self.category,
            "page_number": self.page_number,
            "page_numbers": self.page_numbers,
            "section_hierarchy": self.section_hierarchy,
            "chunk_strategy": self.chunk_strategy,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }


@dataclass
class Chunk:
    """A single text chunk with its metadata."""

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    strategy_name: str = "base"

    @abstractmethod
    def chunk(
        self,
        document: ParsedDocument,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """
        Split a parsed document into chunks.

        Args:
            document: The parsed PDF document.
            metadata: Extracted document metadata.

        Returns:
            List of Chunk objects with populated metadata.
        """
        ...

    def _build_base_metadata(
        self,
        metadata: DocumentMetadata,
        page_number: int = 0,
        page_numbers: list[int] | None = None,
        section_hierarchy: list[str] | None = None,
        chunk_index: int = 0,
        token_count: int = 0,
        text: str = "",
    ) -> ChunkMetadata:
        """Helper to construct ChunkMetadata from document metadata."""
        return ChunkMetadata(
            document_title=metadata.title,
            file_name=metadata.file_name,
            file_hash=metadata.file_hash,
            circular_number=metadata.circular_number,
            date=metadata.date,
            category=metadata.category,
            page_number=page_number,
            page_numbers=page_numbers or ([page_number] if page_number else []),
            section_hierarchy=section_hierarchy or [],
            chunk_strategy=self.strategy_name,
            chunk_index=chunk_index,
            token_count=token_count,
            char_count=len(text),
        )

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens using tiktoken (cl100k_base encoding)."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Rough fallback: ~4 chars per token
            return len(text) // 4
