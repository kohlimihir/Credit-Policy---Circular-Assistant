"""
Strategy B: Recursive Character Text Splitter with Overlap

Uses LangChain's RecursiveCharacterTextSplitter with a hierarchy of
separators (\\n\\n → \\n → '. ' → ' ') and configurable overlap.
This is the standard "good-enough" chunking approach.
"""

from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.chunking.base import BaseChunker, Chunk
from src.ingestion.metadata_extractor import DocumentMetadata
from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """Recursive text splitter with overlap (Strategy B)."""

    strategy_name = "recursive_overlap"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            # Use tiktoken-based length function
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=lambda t: len(self._enc.encode(t)),
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
                keep_separator=True,
            )
        except ImportError:
            self._enc = None
            # Fallback to character-based (~4 chars per token)
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,
                chunk_overlap=chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
                keep_separator=True,
            )
            logger.warning("tiktoken not installed — using char-based size")

    def chunk(
        self,
        document: ParsedDocument,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """
        Split document using recursive character text splitting.

        Processes each page individually to preserve page-number provenance,
        then applies the recursive splitter.
        """
        chunks: list[Chunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue

            # Split this page's text
            text_splits = self._splitter.split_text(page.text.strip())

            for split_text in text_splits:
                if not split_text.strip():
                    continue

                token_count = self.count_tokens(split_text)

                # Find section hierarchy for this chunk
                section_hierarchy = self._find_section_hierarchy(
                    split_text, page.page_number, metadata
                )

                chunk_meta = self._build_base_metadata(
                    metadata=metadata,
                    page_number=page.page_number,
                    section_hierarchy=section_hierarchy,
                    chunk_index=len(chunks),
                    token_count=token_count,
                    text=split_text,
                )
                chunks.append(Chunk(text=split_text, metadata=chunk_meta))

        logger.info(
            "RecursiveChunker: %s → %d chunks (size=%d, overlap=%d)",
            document.file_name,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    @staticmethod
    def _find_section_hierarchy(
        text: str,
        page_number: int,
        metadata: DocumentMetadata,
    ) -> list[str]:
        """
        Find the section hierarchy for a chunk based on its page position.

        Looks at the document's detected sections and finds the most recent
        section headings at each level that precede this chunk's page.
        """
        if not metadata.sections:
            return []

        # Find the most recent heading at each level on or before this page
        hierarchy: dict[int, str] = {}
        for section in metadata.sections:
            if section.page_number <= page_number:
                hierarchy[section.level] = section.heading
            else:
                break  # sections are sorted by offset

        # Build ordered hierarchy: level 1, then 2, then 3
        return [
            hierarchy[level]
            for level in sorted(hierarchy.keys())
        ]
