"""
Strategy A: Fixed 512-Token Chunker

Splits document text into uniform chunks of exactly `chunk_size` tokens
with no overlap.  This is the naive baseline for comparison.
"""

from __future__ import annotations

import logging

from src.ingestion.chunking.base import BaseChunker, Chunk
from src.ingestion.metadata_extractor import DocumentMetadata
from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


class FixedChunker(BaseChunker):
    """Fixed-size token chunker (baseline strategy)."""

    strategy_name = "fixed_512"

    def __init__(self, chunk_size: int = 512) -> None:
        self.chunk_size = chunk_size
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._enc = None
            logger.warning("tiktoken not installed — using char-based fallback")

    def chunk(
        self,
        document: ParsedDocument,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """Split into fixed-size token chunks, no overlap."""

        # Build a list of (text, page_number) so we can track page provenance
        page_texts: list[tuple[str, int]] = []
        for page in document.pages:
            if page.text.strip():
                page_texts.append((page.text.strip(), page.page_number))

        if not page_texts:
            return []

        # Tokenize or split the full concatenated text
        full_text = "\n\n".join(t for t, _ in page_texts)

        if self._enc is not None:
            tokens = self._enc.encode(full_text)
            chunks = self._split_tokens(tokens, page_texts, metadata)
        else:
            chunks = self._split_chars(full_text, page_texts, metadata)

        logger.info(
            "FixedChunker: %s → %d chunks (size=%d tokens)",
            document.file_name,
            len(chunks),
            self.chunk_size,
        )
        return chunks

    def _split_tokens(
        self,
        tokens: list[int],
        page_texts: list[tuple[str, int]],
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """Split token sequence into fixed-size chunks."""
        chunks: list[Chunk] = []
        total = len(tokens)

        for i in range(0, total, self.chunk_size):
            token_slice = tokens[i : i + self.chunk_size]
            text = self._enc.decode(token_slice)
            token_count = len(token_slice)

            # Estimate which page(s) this chunk falls on
            page_number = self._estimate_page(text, page_texts)

            chunk_meta = self._build_base_metadata(
                metadata=metadata,
                page_number=page_number,
                chunk_index=len(chunks),
                token_count=token_count,
                text=text,
            )
            chunks.append(Chunk(text=text, metadata=chunk_meta))

        return chunks

    def _split_chars(
        self,
        full_text: str,
        page_texts: list[tuple[str, int]],
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """Fallback: split by character count (~4 chars per token)."""
        char_size = self.chunk_size * 4
        chunks: list[Chunk] = []

        for i in range(0, len(full_text), char_size):
            text = full_text[i : i + char_size]
            token_count = self.count_tokens(text)
            page_number = self._estimate_page(text, page_texts)

            chunk_meta = self._build_base_metadata(
                metadata=metadata,
                page_number=page_number,
                chunk_index=len(chunks),
                token_count=token_count,
                text=text,
            )
            chunks.append(Chunk(text=text, metadata=chunk_meta))

        return chunks

    @staticmethod
    def _estimate_page(
        chunk_text: str, page_texts: list[tuple[str, int]]
    ) -> int:
        """Estimate which page a chunk primarily belongs to by overlap."""
        if not page_texts:
            return 0

        # Simple heuristic: check which page text has the most overlap
        # with the first 200 chars of the chunk
        snippet = chunk_text[:200]
        best_page = page_texts[0][1]
        best_overlap = 0

        for page_text, page_num in page_texts:
            # Count shared words as a cheap overlap proxy
            chunk_words = set(snippet.lower().split())
            page_words = set(page_text[:500].lower().split())
            overlap = len(chunk_words & page_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page_num

        return best_page
