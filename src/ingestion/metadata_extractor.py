"""
Metadata Extractor — pulls structured metadata from parsed RBI/NBFC PDF documents.

Extracts:
  - Document title
  - Circular / master direction number (e.g. RBI/2023-24/XX or DoR.FIN.REC.XX)
  - Date of issuance
  - Document category (master direction, circular, notification, etc.)
  - Section hierarchy for structural chunking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Structured metadata extracted from an RBI / NBFC document."""

    file_name: str
    file_hash: str
    title: str = ""
    circular_number: str = ""
    date: str = ""
    category: str = "unknown"
    issuing_department: str = ""
    applicable_to: list[str] = field(default_factory=list)
    total_pages: int = 0
    sections: list[SectionInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "title": self.title,
            "circular_number": self.circular_number,
            "date": self.date,
            "category": self.category,
            "issuing_department": self.issuing_department,
            "applicable_to": self.applicable_to,
            "total_pages": self.total_pages,
            "sections": [s.to_dict() for s in self.sections],
        }


@dataclass
class SectionInfo:
    """Detected section within a document."""

    heading: str
    level: int  # 1 = chapter, 2 = section, 3 = sub-section
    page_number: int
    start_char_offset: int  # character offset in full_text
    parent_heading: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "level": self.level,
            "page_number": self.page_number,
            "start_char_offset": self.start_char_offset,
            "parent_heading": self.parent_heading,
        }


class MetadataExtractor:
    """
    Extracts structured metadata from parsed RBI / NBFC documents.

    Uses regex patterns calibrated to RBI's formatting conventions:
    - Circular numbers: RBI/YYYY-YY/NNN, DNBR.PD.CC.No.NNN
    - Section patterns: Chapter I, Section 5, (a), (i), etc.
    - Date formats: Month DD, YYYY and DD.MM.YYYY
    """

    # ── Circular number patterns ─────────────────────────────────────────
    CIRCULAR_PATTERNS = [
        # RBI/2023-24/XX format
        re.compile(r"RBI/\d{4}-\d{2}/\d+", re.IGNORECASE),
        # Department reference: DoR.FIN.REC.XX/XX.XX.XXX/YYYY-YY
        re.compile(
            r"[A-Z]{2,5}(?:\.[A-Z]{2,5}){1,4}(?:\.No)?\.?\d*[/\.\-]\d+[/\.\-][\d\-]+",
            re.IGNORECASE,
        ),
        # DNBR / DNBS legacy format
        re.compile(r"DNBR?\.\w+\.\w+\.No\.\d+", re.IGNORECASE),
        # DOR format
        re.compile(r"DOR\.\w+\.REC\.\d+/\d+\.\d+\.\d+/\d{4}-\d{2}", re.IGNORECASE),
    ]

    # ── Date patterns ────────────────────────────────────────────────────
    DATE_PATTERNS = [
        # January 15, 2024
        (
            re.compile(
                r"(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2},?\s+\d{4}",
                re.IGNORECASE,
            ),
            "%B %d, %Y",
        ),
        # 15.01.2024
        (re.compile(r"\d{2}\.\d{2}\.\d{4}"), "%d.%m.%Y"),
        # 2024-01-15
        (re.compile(r"\d{4}-\d{2}-\d{2}"), "%Y-%m-%d"),
    ]

    # ── Section heading patterns (ordered by hierarchical level) ─────────
    SECTION_PATTERNS = [
        # Level 1: Chapter / Part
        (
            re.compile(
                r"^(?:CHAPTER|PART)\s+[IVXLCDM]+[\.\s\-:]+(.+?)$",
                re.IGNORECASE | re.MULTILINE,
            ),
            1,
        ),
        (
            re.compile(
                r"^(?:CHAPTER|PART)\s+\d+[\.\s\-:]+(.+?)$",
                re.IGNORECASE | re.MULTILINE,
            ),
            1,
        ),
        # Level 2: Section / Article / numbered headings
        (
            re.compile(
                r"^(?:Section|Article)\s+\d+[\.\s\-:]+(.+?)$",
                re.IGNORECASE | re.MULTILINE,
            ),
            2,
        ),
        (
            re.compile(r"^\d+\.\s+[A-Z][A-Za-z\s\-&]{5,}$", re.MULTILINE),
            2,
        ),
        # Level 3: Sub-sections (a), (b), (i), (ii)
        (
            re.compile(
                r"^\s*\([a-z]\)\s+[A-Z].{10,}$",
                re.MULTILINE,
            ),
            3,
        ),
        (
            re.compile(
                r"^\s*\([ivx]+\)\s+.{10,}$",
                re.IGNORECASE | re.MULTILINE,
            ),
            3,
        ),
    ]

    # ── Category detection ───────────────────────────────────────────────
    CATEGORY_KEYWORDS = {
        "master direction": "master_direction",
        "master circular": "master_circular",
        "circular": "circular",
        "notification": "notification",
        "guidelines": "guidelines",
        "advisory": "advisory",
        "press release": "press_release",
        "fair practices code": "fair_practices_code",
    }

    def extract(self, doc: ParsedDocument) -> DocumentMetadata:
        """
        Extract structured metadata from a ParsedDocument.

        Args:
            doc: ParsedDocument from PDFParser.

        Returns:
            DocumentMetadata with extracted fields.
        """
        full_text = doc.full_text
        # Use first 3 pages for metadata detection (title, circular no, date)
        header_text = "\n".join(
            p.text for p in doc.pages[:3] if p.text.strip()
        )

        title = self._extract_title(doc, header_text)
        circular_number = self._extract_circular_number(header_text)
        date = self._extract_date(header_text)
        category = self._detect_category(title, header_text)
        department = self._extract_department(header_text)
        applicable_to = self._extract_applicability(header_text)
        sections = self._extract_sections(doc)

        metadata = DocumentMetadata(
            file_name=doc.file_name,
            file_hash=doc.file_hash,
            title=title,
            circular_number=circular_number,
            date=date,
            category=category,
            issuing_department=department,
            applicable_to=applicable_to,
            total_pages=doc.total_pages,
            sections=sections,
        )

        logger.info(
            "Extracted metadata for %s: title='%s', circular='%s', date='%s', "
            "category='%s', sections=%d",
            doc.file_name,
            title[:60],
            circular_number,
            date,
            category,
            len(sections),
        )
        return metadata

    # ── Private extraction methods ───────────────────────────────────────

    def _extract_title(self, doc: ParsedDocument, header_text: str) -> str:
        """Extract document title from PDF metadata or first page content."""
        # Try PDF metadata first
        pdf_title = doc.raw_metadata.get("title", "").strip()
        if pdf_title and len(pdf_title) > 10 and not pdf_title.startswith("Microsoft"):
            return pdf_title

        # Look for "Master Direction" or "Circular" line patterns
        title_patterns = [
            re.compile(
                r"Master\s+Direction\s*[-–—:]\s*(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:Circular|Notification|Guidelines)\s+on\s+(.+?)(?:\n|$)",
                re.IGNORECASE,
            ),
        ]
        for pattern in title_patterns:
            match = pattern.search(header_text)
            if match:
                return f"Master Direction - {match.group(1).strip()}" if "master" in pattern.pattern.lower() else match.group(1).strip()

        # Fallback: use the first non-trivial line from page 1
        if doc.pages:
            lines = doc.pages[0].text.split("\n")
            for line in lines:
                clean = line.strip()
                if len(clean) > 15 and not re.match(r"^[\d\.\-/\s]+$", clean):
                    return clean[:200]

        return doc.file_name.replace(".pdf", "").replace("_", " ")

    def _extract_circular_number(self, header_text: str) -> str:
        """Extract RBI circular / master direction reference number."""
        for pattern in self.CIRCULAR_PATTERNS:
            match = pattern.search(header_text)
            if match:
                return match.group(0).strip()
        return ""

    def _extract_date(self, header_text: str) -> str:
        """Extract date of issuance in ISO format."""
        for pattern, fmt in self.DATE_PATTERNS:
            match = pattern.search(header_text)
            if match:
                try:
                    date_str = match.group(0).replace(",", ", ")
                    # Normalize spacing
                    date_str = re.sub(r"\s+", " ", date_str).strip()
                    parsed = datetime.strptime(date_str, fmt)
                    return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        return ""

    def _detect_category(self, title: str, header_text: str) -> str:
        """Classify document type based on title and header content."""
        combined = f"{title} {header_text}".lower()
        for keyword, category in self.CATEGORY_KEYWORDS.items():
            if keyword in combined:
                return category
        return "unknown"

    @staticmethod
    def _extract_department(header_text: str) -> str:
        """Extract issuing RBI department."""
        dept_patterns = [
            re.compile(
                r"Department\s+of\s+(?:Regulation|Supervision|"
                r"Non-Banking\s+Supervision|Payment\s+and\s+Settlement\s+Systems)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:DoR|DoS|DNBS|DPSS|DOR|FIDD)\b",
                re.IGNORECASE,
            ),
        ]
        for pattern in dept_patterns:
            match = pattern.search(header_text)
            if match:
                return match.group(0).strip()
        return ""

    @staticmethod
    def _extract_applicability(header_text: str) -> list[str]:
        """Extract entities the document applies to (NBFC-ML, banks, etc.)."""
        entity_patterns = [
            r"NBFC[-\s]?(?:ML|BL|SI|MFI|P2P|AA|ND|D|ND-SI)",
            r"All\s+(?:NBFCs|Banks|Scheduled\s+Commercial\s+Banks)",
            r"Non-Banking\s+Financial\s+Companies?",
            r"Housing\s+Finance\s+Companies?",
            r"Micro\s*Finance\s+Institutions?",
        ]
        entities: list[str] = []
        for pattern in entity_patterns:
            matches = re.findall(pattern, header_text, re.IGNORECASE)
            entities.extend(m.strip() for m in matches)
        return list(set(entities))

    def _extract_sections(self, doc: ParsedDocument) -> list[SectionInfo]:
        """
        Detect section headings throughout the document.
        Returns a flat list of SectionInfo with hierarchical level tags.
        """
        sections: list[SectionInfo] = []
        full_text = doc.full_text
        char_offset = 0
        current_parents: dict[int, str] = {}  # level -> heading

        for page in doc.pages:
            page_text = page.text
            for pattern, level in self.SECTION_PATTERNS:
                for match in pattern.finditer(page_text):
                    heading = match.group(0).strip()
                    abs_offset = char_offset + match.start()

                    # Update parent tracking
                    current_parents[level] = heading
                    parent = current_parents.get(level - 1, "")

                    sections.append(
                        SectionInfo(
                            heading=heading,
                            level=level,
                            page_number=page.page_number,
                            start_char_offset=abs_offset,
                            parent_heading=parent,
                        )
                    )
            # Account for page separator ("\n\n")
            char_offset += len(page_text) + 2

        # Sort by document order
        sections.sort(key=lambda s: s.start_char_offset)
        return sections
