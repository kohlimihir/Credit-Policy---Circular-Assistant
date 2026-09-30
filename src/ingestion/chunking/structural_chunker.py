"""
Strategy C: Section-Aware Structural Chunker

The most sophisticated strategy — respects document structure by splitting at
section boundaries.  Never breaks mid-section.  If a section exceeds the max
token limit, it is sub-chunked at paragraph boundaries.

This strategy produces the highest-quality chunks for regulatory documents
because sections are the natural semantic units of RBI master directions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.ingestion.chunking.base import BaseChunker, Chunk
from src.ingestion.metadata_extractor import DocumentMetadata, SectionInfo
from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class _SectionBlock:
    """Internal representation of a document section with text."""

    heading: str
    level: int
    page_number: int
    hierarchy: list[str]
    text: str
    token_count: int = 0


class StructuralChunker(BaseChunker):
    """Section-aware structural chunker (Strategy C)."""

    strategy_name = "structural"

    def __init__(self, max_chunk_size: int = 1024) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(
        self,
        document: ParsedDocument,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """
        Split document respecting section boundaries.

        Algorithm:
        1. Build a full text with page markers.
        2. Use detected section headings to segment the text.
        3. For each section:
           - If ≤ max_chunk_size tokens → single chunk.
           - If > max_chunk_size → sub-split at paragraph boundaries.
        4. Attach full section hierarchy as metadata.
        """
        if not metadata.sections:
            # Fallback: if no sections detected, split by page
            logger.warning(
                "No sections detected in %s — falling back to page-based splitting",
                document.file_name,
            )
            return self._fallback_page_chunks(document, metadata)

        # Build full text and page-offset map
        full_text, page_offsets = self._build_full_text_with_offsets(document)

        # Segment text into section blocks
        section_blocks = self._segment_into_sections(
            full_text, metadata.sections, page_offsets
        )

        # Convert section blocks into chunks (with sub-chunking for large sections)
        chunks: list[Chunk] = []
        for block in section_blocks:
            if not block.text.strip():
                continue

            block.token_count = self.count_tokens(block.text)

            if block.token_count <= self.max_chunk_size:
                # Section fits in one chunk
                chunk_meta = self._build_base_metadata(
                    metadata=metadata,
                    page_number=block.page_number,
                    section_hierarchy=block.hierarchy,
                    chunk_index=len(chunks),
                    token_count=block.token_count,
                    text=block.text,
                )
                chunks.append(Chunk(text=block.text, metadata=chunk_meta))
            else:
                # Sub-chunk large sections at paragraph boundaries
                sub_chunks = self._sub_chunk_section(block, metadata, len(chunks))
                chunks.extend(sub_chunks)

        logger.info(
            "StructuralChunker: %s → %d chunks (max_size=%d tokens, sections=%d)",
            document.file_name,
            len(chunks),
            self.max_chunk_size,
            len(metadata.sections),
        )
        return chunks

    # ── Core segmentation logic ──────────────────────────────────────────

    @staticmethod
    def _build_full_text_with_offsets(
        document: ParsedDocument,
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """
        Build concatenated full text and track (start_offset, end_offset, page_num)
        for each page so we can map character offsets back to page numbers.
        """
        parts: list[str] = []
        offsets: list[tuple[int, int, int]] = []
        current_offset = 0

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue
            start = current_offset
            parts.append(text)
            current_offset += len(text) + 2  # +2 for the "\n\n" separator
            offsets.append((start, current_offset - 2, page.page_number))

        full_text = "\n\n".join(parts)
        return full_text, offsets

    def _segment_into_sections(
        self,
        full_text: str,
        sections: list[SectionInfo],
        page_offsets: list[tuple[int, int, int]],
    ) -> list[_SectionBlock]:
        """
        Segment full text using detected section boundaries.

        Each section's text spans from its heading position to the start
        of the next section.
        """
        blocks: list[_SectionBlock] = []

        for i, section in enumerate(sections):
            start = section.start_char_offset
            # End at the next section's start, or end of text
            if i + 1 < len(sections):
                end = sections[i + 1].start_char_offset
            else:
                end = len(full_text)

            text = full_text[start:end].strip()
            if not text:
                continue

            # Build hierarchy
            hierarchy = self._build_hierarchy(section, sections[:i])

            # Determine page number
            page_number = self._offset_to_page(start, page_offsets)

            blocks.append(
                _SectionBlock(
                    heading=section.heading,
                    level=section.level,
                    page_number=page_number,
                    hierarchy=hierarchy,
                    text=text,
                )
            )

        # Handle text before the first section (preamble)
        if sections and sections[0].start_char_offset > 0:
            preamble = full_text[: sections[0].start_char_offset].strip()
            if preamble and len(preamble) > 50:
                preamble_page = self._offset_to_page(0, page_offsets)
                blocks.insert(
                    0,
                    _SectionBlock(
                        heading="Preamble",
                        level=0,
                        page_number=preamble_page,
                        hierarchy=["Preamble"],
                        text=preamble,
                    ),
                )

        return blocks

    def _sub_chunk_section(
        self,
        block: _SectionBlock,
        metadata: DocumentMetadata,
        base_chunk_index: int,
    ) -> list[Chunk]:
        """
        Sub-chunk a large section at paragraph boundaries.

        Strategy: split on double newlines first, then single newlines.
        Group paragraphs until reaching max_chunk_size.
        """
        paragraphs = re.split(r"\n\n+", block.text)
        if len(paragraphs) <= 1:
            # If no paragraph breaks, split on single newlines
            paragraphs = block.text.split("\n")

        chunks: list[Chunk] = []
        current_text_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > self.max_chunk_size and current_text_parts:
                # Emit current accumulated chunk
                chunk_text = "\n\n".join(current_text_parts)
                chunk_meta = self._build_base_metadata(
                    metadata=metadata,
                    page_number=block.page_number,
                    section_hierarchy=block.hierarchy,
                    chunk_index=base_chunk_index + len(chunks),
                    token_count=current_tokens,
                    text=chunk_text,
                )
                chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))
                current_text_parts = []
                current_tokens = 0

            current_text_parts.append(para)
            current_tokens += para_tokens

        # Emit remaining
        if current_text_parts:
            chunk_text = "\n\n".join(current_text_parts)
            chunk_meta = self._build_base_metadata(
                metadata=metadata,
                page_number=block.page_number,
                section_hierarchy=block.hierarchy,
                chunk_index=base_chunk_index + len(chunks),
                token_count=self.count_tokens(chunk_text),
                text=chunk_text,
            )
            chunks.append(Chunk(text=chunk_text, metadata=chunk_meta))

        return chunks

    def _fallback_page_chunks(
        self,
        document: ParsedDocument,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        """Fallback: one chunk per page when no sections are detected."""
        chunks: list[Chunk] = []
        for page in document.pages:
            if not page.text.strip():
                continue

            token_count = self.count_tokens(page.text)
            chunk_meta = self._build_base_metadata(
                metadata=metadata,
                page_number=page.page_number,
                chunk_index=len(chunks),
                token_count=token_count,
                text=page.text,
            )
            chunks.append(Chunk(text=page.text.strip(), metadata=chunk_meta))
        return chunks

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_hierarchy(
        current: SectionInfo, preceding: list[SectionInfo]
    ) -> list[str]:
        """Build the full section hierarchy path for a given section."""
        hierarchy: dict[int, str] = {}

        # Walk preceding sections to find parents
        for sec in preceding:
            if sec.level < current.level:
                hierarchy[sec.level] = sec.heading

        hierarchy[current.level] = current.heading

        return [hierarchy[lvl] for lvl in sorted(hierarchy.keys())]

    @staticmethod
    def _offset_to_page(
        char_offset: int,
        page_offsets: list[tuple[int, int, int]],
    ) -> int:
        """Map a character offset in the full text to a page number."""
        for start, end, page_num in page_offsets:
            if start <= char_offset <= end:
                return page_num
        # Fallback to last page
        return page_offsets[-1][2] if page_offsets else 1
