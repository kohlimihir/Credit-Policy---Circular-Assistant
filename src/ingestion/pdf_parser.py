"""
PDF Parser — extracts text, tables, and structural information from PDF documents.

Uses PyMuPDF (fitz) as primary extractor with pdfplumber fallback for
table-heavy pages.  Each page is returned as a structured dict preserving
page number, raw text, detected tables, and layout hints.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Structured representation of a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    has_images: bool = False
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class ParsedDocument:
    """Complete parsed PDF document with metadata."""

    file_path: str
    file_name: str
    file_hash: str
    total_pages: int
    pages: list[ParsedPage]
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenated text across all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class PDFParser:
    """
    Multi-strategy PDF parser.

    Strategy:
    1. PyMuPDF for fast text extraction with layout detection.
    2. pdfplumber fallback for pages with < 50 chars (likely scanned or table-only).
    3. Table extraction always via pdfplumber for accuracy.
    """

    MIN_TEXT_CHARS = 50  # below this, page is likely scanned / image-only

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a single PDF file into a structured ParsedDocument.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            ParsedDocument with per-page text, tables, and metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        file_hash = self._compute_hash(pdf_path)
        logger.info("Parsing PDF: %s (hash: %s)", pdf_path.name, file_hash[:12])

        pages = self._extract_with_pymupdf(pdf_path)
        raw_metadata = self._extract_pdf_metadata(pdf_path)

        # Fallback for pages with very little text
        pages = self._fallback_pdfplumber(pdf_path, pages)

        # Always try table extraction
        pages = self._extract_tables(pdf_path, pages)

        return ParsedDocument(
            file_path=str(pdf_path),
            file_name=pdf_path.name,
            file_hash=file_hash,
            total_pages=len(pages),
            pages=pages,
            raw_metadata=raw_metadata,
        )

    def parse_directory(self, dir_path: str | Path) -> list[ParsedDocument]:
        """Parse all PDFs in a directory."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        pdf_files = sorted(dir_path.glob("**/*.pdf"))
        logger.info("Found %d PDF files in %s", len(pdf_files), dir_path)

        documents: list[ParsedDocument] = []
        for pdf_file in pdf_files:
            try:
                doc = self.parse(pdf_file)
                documents.append(doc)
                logger.info(
                    "  ✓ %s — %d pages, %d chars",
                    pdf_file.name,
                    doc.total_pages,
                    sum(p.char_count for p in doc.pages),
                )
            except Exception:
                logger.exception("  ✗ Failed to parse: %s", pdf_file.name)

        logger.info(
            "Parsed %d / %d PDFs successfully", len(documents), len(pdf_files)
        )
        return documents

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """SHA-256 hash for deduplication."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _extract_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
        """Extract built-in PDF metadata (title, author, date, etc.)."""
        try:
            with fitz.open(str(pdf_path)) as doc:
                meta = doc.metadata or {}
                return {k: v for k, v in meta.items() if v}
        except Exception:
            logger.warning("Could not extract metadata from %s", pdf_path.name)
            return {}

    @staticmethod
    def _extract_with_pymupdf(pdf_path: Path) -> list[ParsedPage]:
        """Primary extraction: PyMuPDF with layout-aware text."""
        pages: list[ParsedPage] = []
        with fitz.open(str(pdf_path)) as doc:
            for page_idx, page in enumerate(doc):
                # Use "text" mode which handles columns reasonably well
                text = page.get_text("text")

                # Clean up excessive whitespace but preserve paragraph breaks
                lines = text.split("\n")
                cleaned_lines: list[str] = []
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        cleaned_lines.append(stripped)
                    elif cleaned_lines and cleaned_lines[-1] != "":
                        cleaned_lines.append("")  # preserve paragraph breaks

                cleaned_text = "\n".join(cleaned_lines).strip()
                has_images = len(page.get_images()) > 0

                pages.append(
                    ParsedPage(
                        page_number=page_idx + 1,
                        text=cleaned_text,
                        has_images=has_images,
                    )
                )
        return pages

    def _fallback_pdfplumber(
        self, pdf_path: Path, pages: list[ParsedPage]
    ) -> list[ParsedPage]:
        """Re-extract text with pdfplumber for pages that PyMuPDF couldn't handle."""
        sparse_pages = [
            i for i, p in enumerate(pages) if p.char_count < self.MIN_TEXT_CHARS
        ]
        if not sparse_pages:
            return pages

        logger.info(
            "Using pdfplumber fallback for %d sparse pages in %s",
            len(sparse_pages),
            pdf_path.name,
        )

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for idx in sparse_pages:
                    if idx < len(pdf.pages):
                        plumber_page = pdf.pages[idx]
                        text = plumber_page.extract_text() or ""
                        if len(text) > pages[idx].char_count:
                            pages[idx].text = text.strip()
                            pages[idx].char_count = len(text.strip())
        except Exception:
            logger.warning("pdfplumber fallback failed for %s", pdf_path.name)

        return pages

    @staticmethod
    def _extract_tables(
        pdf_path: Path, pages: list[ParsedPage]
    ) -> list[ParsedPage]:
        """Extract tables from all pages using pdfplumber."""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_idx, plumber_page in enumerate(pdf.pages):
                    if page_idx < len(pages):
                        tables = plumber_page.extract_tables() or []
                        # Convert None cells to empty strings
                        cleaned_tables = [
                            [
                                [cell if cell is not None else "" for cell in row]
                                for row in table
                            ]
                            for table in tables
                        ]
                        pages[page_idx].tables = cleaned_tables
        except Exception:
            logger.warning("Table extraction failed for %s", pdf_path.name)

        return pages
