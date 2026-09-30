"""
Tests for chunking strategies.

Verifies that all three chunkers produce valid chunks with correct metadata.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.ingestion.pdf_parser import ParsedDocument, ParsedPage
from src.ingestion.metadata_extractor import DocumentMetadata, SectionInfo
from src.ingestion.chunking import FixedChunker, RecursiveChunker, StructuralChunker


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_document() -> ParsedDocument:
    """Create a sample parsed document for testing."""
    pages = [
        ParsedPage(
            page_number=1,
            text=(
                "CHAPTER I - PRELIMINARY\n\n"
                "1. Short Title and Commencement\n\n"
                "These directions shall be known as the 'Master Direction – "
                "Non-Banking Financial Company – Systemically Important Non-Deposit "
                "taking Company and Deposit taking Company (Reserve Bank) Directions, 2016'.\n\n"
                "2. Applicability\n\n"
                "These directions shall be applicable to all Non-Banking Financial "
                "Companies as defined in clause (f) of section 45-I of the Reserve "
                "Bank of India Act, 1934. " * 10
            ),
        ),
        ParsedPage(
            page_number=2,
            text=(
                "CHAPTER II - CAPITAL REQUIREMENTS\n\n"
                "3. Capital to Risk-weighted Assets Ratio (CRAR)\n\n"
                "Every NBFC-ND-SI shall maintain a minimum Capital to Risk-weighted "
                "Assets Ratio (CRAR) of 15 per cent. The Tier I capital shall not be "
                "less than 10 per cent. " * 10 + "\n\n"
                "4. Tier I Capital\n\n"
                "Tier I Capital means owned fund as reduced by investment in shares "
                "of other NBFCs and in shares, debentures, bonds, outstanding loans "
                "and advances. " * 8
            ),
        ),
        ParsedPage(
            page_number=3,
            text=(
                "CHAPTER III - EXPOSURE NORMS\n\n"
                "5. Single Borrower Exposure\n\n"
                "The exposure to any single borrower shall not exceed 25 per cent of "
                "its owned fund. The Board of Directors may set lower internal limits. " * 10 + "\n\n"
                "6. Group Exposure\n\n"
                "The exposure to any single group of borrowers shall not exceed 40 "
                "per cent of its owned fund. " * 8
            ),
        ),
    ]

    return ParsedDocument(
        file_path="/test/test_doc.pdf",
        file_name="test_doc.pdf",
        file_hash="abc123",
        total_pages=3,
        pages=pages,
    )


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Create sample metadata for testing."""
    return DocumentMetadata(
        file_name="test_doc.pdf",
        file_hash="abc123",
        title="Master Direction - NBFC-ND-SI",
        circular_number="RBI/2023-24/01",
        date="2024-01-15",
        category="master_direction",
        total_pages=3,
        sections=[
            SectionInfo(heading="CHAPTER I - PRELIMINARY", level=1, page_number=1, start_char_offset=0),
            SectionInfo(heading="CHAPTER II - CAPITAL REQUIREMENTS", level=1, page_number=2, start_char_offset=500),
            SectionInfo(heading="CHAPTER III - EXPOSURE NORMS", level=1, page_number=3, start_char_offset=1200),
        ],
    )


# ── Fixed Chunker Tests ─────────────────────────────────────────────────

class TestFixedChunker:
    def test_produces_chunks(self, sample_document, sample_metadata):
        chunker = FixedChunker(chunk_size=128)
        chunks = chunker.chunk(sample_document, sample_metadata)
        assert len(chunks) > 0

    def test_chunk_strategy_name(self, sample_document, sample_metadata):
        chunker = FixedChunker()
        chunks = chunker.chunk(sample_document, sample_metadata)
        for chunk in chunks:
            assert chunk.metadata.chunk_strategy == "fixed_512"

    def test_chunk_has_text(self, sample_document, sample_metadata):
        chunker = FixedChunker(chunk_size=128)
        chunks = chunker.chunk(sample_document, sample_metadata)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_chunk_metadata_populated(self, sample_document, sample_metadata):
        chunker = FixedChunker(chunk_size=128)
        chunks = chunker.chunk(sample_document, sample_metadata)
        for chunk in chunks:
            assert chunk.metadata.document_title == "Master Direction - NBFC-ND-SI"
            assert chunk.metadata.file_name == "test_doc.pdf"
            assert chunk.metadata.chunk_id  # non-empty UUID


# ── Recursive Chunker Tests ─────────────────────────────────────────────

class TestRecursiveChunker:
    def test_produces_chunks(self, sample_document, sample_metadata):
        chunker = RecursiveChunker(chunk_size=128, chunk_overlap=32)
        chunks = chunker.chunk(sample_document, sample_metadata)
        assert len(chunks) > 0

    def test_chunk_strategy_name(self, sample_document, sample_metadata):
        chunker = RecursiveChunker()
        chunks = chunker.chunk(sample_document, sample_metadata)
        for chunk in chunks:
            assert chunk.metadata.chunk_strategy == "recursive_overlap"

    def test_preserves_page_numbers(self, sample_document, sample_metadata):
        chunker = RecursiveChunker(chunk_size=128, chunk_overlap=32)
        chunks = chunker.chunk(sample_document, sample_metadata)
        page_numbers = {c.metadata.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers


# ── Structural Chunker Tests ────────────────────────────────────────────

class TestStructuralChunker:
    def test_produces_chunks(self, sample_document, sample_metadata):
        chunker = StructuralChunker(max_chunk_size=256)
        chunks = chunker.chunk(sample_document, sample_metadata)
        assert len(chunks) > 0

    def test_chunk_strategy_name(self, sample_document, sample_metadata):
        chunker = StructuralChunker()
        chunks = chunker.chunk(sample_document, sample_metadata)
        for chunk in chunks:
            assert chunk.metadata.chunk_strategy == "structural"

    def test_section_hierarchy_populated(self, sample_document, sample_metadata):
        chunker = StructuralChunker(max_chunk_size=256)
        chunks = chunker.chunk(sample_document, sample_metadata)
        has_hierarchy = any(len(c.metadata.section_hierarchy) > 0 for c in chunks)
        assert has_hierarchy, "At least one chunk should have section hierarchy"

    def test_fallback_on_no_sections(self, sample_document):
        """When no sections are detected, should fall back to page-based chunks."""
        meta_no_sections = DocumentMetadata(
            file_name="test_doc.pdf",
            file_hash="abc123",
            title="Test Document",
            total_pages=3,
            sections=[],  # No sections
        )
        chunker = StructuralChunker()
        chunks = chunker.chunk(sample_document, meta_no_sections)
        assert len(chunks) > 0  # Should still produce chunks via fallback


# ── Cross-Strategy Tests ────────────────────────────────────────────────

class TestCrossStrategy:
    def test_all_strategies_produce_output(self, sample_document, sample_metadata):
        """All three strategies should produce non-empty chunk lists."""
        chunkers = [
            FixedChunker(chunk_size=128),
            RecursiveChunker(chunk_size=128, chunk_overlap=32),
            StructuralChunker(max_chunk_size=256),
        ]
        for chunker in chunkers:
            chunks = chunker.chunk(sample_document, sample_metadata)
            assert len(chunks) > 0, f"{chunker.strategy_name} produced no chunks"

    def test_unique_chunk_ids(self, sample_document, sample_metadata):
        """All chunk IDs should be unique across all strategies."""
        all_ids: set[str] = set()
        chunkers = [
            FixedChunker(chunk_size=128),
            RecursiveChunker(chunk_size=128, chunk_overlap=32),
            StructuralChunker(max_chunk_size=256),
        ]
        for chunker in chunkers:
            chunks = chunker.chunk(sample_document, sample_metadata)
            for chunk in chunks:
                assert chunk.metadata.chunk_id not in all_ids, "Duplicate chunk ID"
                all_ids.add(chunk.metadata.chunk_id)
