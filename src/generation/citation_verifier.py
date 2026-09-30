"""
Citation Verifier — post-generation check ensuring every citation in the
LLM's answer actually exists in the retrieved context.

Parses citations from the generated text, matches them against the
retrieved chunks, and strips any citation that cannot be verified.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A parsed citation from the generated answer."""

    raw_text: str  # The full citation string as it appears in the answer
    document_title: str
    section: str
    page: int | None
    verified: bool = False
    matched_chunk_id: str = ""


class CitationVerifier:
    """
    Verifies that citations in the generated answer match retrieved context.

    Process:
    1. Parse all [Document, Section, Page N] citations from the answer.
    2. For each citation, check if a retrieved chunk matches on:
       - Document title (fuzzy substring match)
       - Section heading (fuzzy substring match)
       - Page number (exact match, if available)
    3. Strip unverified citations or mark them as unverified.
    """

    # Regex to capture citations in [Document, Section, Page N] format
    CITATION_PATTERN = re.compile(
        r"\[([^\]]+?)(?:,\s*(?:Section|Chapter|Part|Paragraph)?\s*"
        r"([^\],]+?))?(?:,\s*[Pp]age\s*(\d+))?\]"
    )

    def verify_and_clean(
        self,
        answer: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Verify citations in the answer against retrieved chunks.

        Args:
            answer: The LLM-generated answer text.
            retrieved_chunks: The chunks that were provided as context.

        Returns:
            Dict with:
                - 'cleaned_answer': str with unverified citations marked
                - 'citations': list of Citation dicts
                - 'verified_count': int
                - 'unverified_count': int
                - 'total_citations': int
        """
        citations = self._parse_citations(answer)

        if not citations:
            logger.info("No citations found in the answer")
            return {
                "cleaned_answer": answer,
                "citations": [],
                "verified_count": 0,
                "unverified_count": 0,
                "total_citations": 0,
            }

        # Verify each citation against retrieved chunks
        for citation in citations:
            citation.verified, citation.matched_chunk_id = self._verify_citation(
                citation, retrieved_chunks
            )

        verified = sum(1 for c in citations if c.verified)
        unverified = sum(1 for c in citations if not c.verified)

        # Clean the answer: mark unverified citations
        cleaned_answer = self._clean_answer(answer, citations)

        logger.info(
            "Citation verification: %d verified, %d unverified out of %d total",
            verified,
            unverified,
            len(citations),
        )

        return {
            "cleaned_answer": cleaned_answer,
            "citations": [
                {
                    "raw_text": c.raw_text,
                    "document_title": c.document_title,
                    "section": c.section,
                    "page": c.page,
                    "verified": c.verified,
                    "matched_chunk_id": c.matched_chunk_id,
                }
                for c in citations
            ],
            "verified_count": verified,
            "unverified_count": unverified,
            "total_citations": len(citations),
        }

    def _parse_citations(self, text: str) -> list[Citation]:
        """Parse all citations from the answer text."""
        citations: list[Citation] = []
        seen_raw: set[str] = set()

        for match in self.CITATION_PATTERN.finditer(text):
            raw = match.group(0)
            if raw in seen_raw:
                continue
            seen_raw.add(raw)

            doc_title = (match.group(1) or "").strip()
            section = (match.group(2) or "").strip()
            page_str = match.group(3)
            page = int(page_str) if page_str else None

            citations.append(
                Citation(
                    raw_text=raw,
                    document_title=doc_title,
                    section=section,
                    page=page,
                )
            )

        return citations

    def _verify_citation(
        self,
        citation: Citation,
        chunks: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        """
        Check if a citation matches any retrieved chunk.

        Returns (is_verified, matched_chunk_id).
        """
        for chunk in chunks:
            chunk_title = chunk.get("document_title", "")
            chunk_sections = chunk.get("section_hierarchy", [])
            chunk_page = chunk.get("page_number", 0)
            chunk_id = chunk.get("chunk_id", "")

            # Document title match (case-insensitive substring)
            title_match = self._fuzzy_match(citation.document_title, chunk_title)
            if not title_match:
                continue

            # Section match (if citation has a section)
            section_match = True
            if citation.section:
                section_match = any(
                    self._fuzzy_match(citation.section, s)
                    for s in chunk_sections
                ) or self._fuzzy_match(citation.section, chunk.get("text", "")[:500])

            # Page match (if citation has a page number)
            page_match = True
            if citation.page is not None and chunk_page > 0:
                # Allow ±1 page tolerance (chunks can span page boundaries)
                page_match = abs(citation.page - chunk_page) <= 1

            if title_match and section_match and page_match:
                return True, chunk_id

        return False, ""

    @staticmethod
    def _fuzzy_match(needle: str, haystack: str) -> bool:
        """Case-insensitive substring match with normalization."""
        if not needle or not haystack:
            return False

        # Normalize whitespace and case
        needle_norm = re.sub(r"\s+", " ", needle.lower().strip())
        haystack_norm = re.sub(r"\s+", " ", haystack.lower().strip())

        # Direct substring match
        if needle_norm in haystack_norm:
            return True

        # Check if significant words overlap (for partial title matches)
        needle_words = set(
            w for w in needle_norm.split() if len(w) > 3
        )
        haystack_words = set(
            w for w in haystack_norm.split() if len(w) > 3
        )

        if not needle_words:
            return False

        overlap = len(needle_words & haystack_words) / len(needle_words)
        return overlap >= 0.6  # at least 60% of significant words match

    @staticmethod
    def _clean_answer(answer: str, citations: list[Citation]) -> str:
        """
        Mark unverified citations in the answer.

        Unverified citations get ' ⚠️ UNVERIFIED' appended.
        """
        cleaned = answer
        for citation in citations:
            if not citation.verified:
                cleaned = cleaned.replace(
                    citation.raw_text,
                    f"{citation.raw_text} ⚠️",
                )
        return cleaned
