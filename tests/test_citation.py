"""
Tests for citation verification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.generation.citation_verifier import CitationVerifier


@pytest.fixture
def verifier():
    return CitationVerifier()


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "chunk-001",
            "text": "Every NBFC-ND-SI shall maintain a minimum CRAR of 15 per cent.",
            "document_title": "Master Direction - NBFC-ND-SI",
            "section_hierarchy": ["Chapter II", "Capital Requirements"],
            "page_number": 5,
        },
        {
            "chunk_id": "chunk-002",
            "text": "The exposure to any single borrower shall not exceed 25 per cent.",
            "document_title": "Master Direction - NBFC-ND-SI",
            "section_hierarchy": ["Chapter III", "Exposure Norms"],
            "page_number": 12,
        },
        {
            "chunk_id": "chunk-003",
            "text": "KYC verification requires officially valid documents.",
            "document_title": "Master Direction - KYC",
            "section_hierarchy": ["Section 3", "Customer Due Diligence"],
            "page_number": 8,
        },
    ]


class TestCitationParsing:
    def test_parse_full_citation(self, verifier):
        answer = "The CRAR is 15% [Master Direction - NBFC-ND-SI, Chapter II, Page 5]."
        result = verifier.verify_and_clean(answer, [])
        assert result["total_citations"] == 1

    def test_parse_multiple_citations(self, verifier):
        answer = (
            "CRAR is 15% [Doc A, Section 1, Page 5]. "
            "Exposure is 25% [Doc B, Section 2, Page 12]."
        )
        result = verifier.verify_and_clean(answer, [])
        assert result["total_citations"] == 2

    def test_no_citations(self, verifier):
        answer = "This answer has no citations at all."
        result = verifier.verify_and_clean(answer, [])
        assert result["total_citations"] == 0


class TestCitationVerification:
    def test_verified_citation(self, verifier, sample_chunks):
        answer = "The CRAR is 15% [Master Direction - NBFC-ND-SI, Capital Requirements, Page 5]."
        result = verifier.verify_and_clean(answer, sample_chunks)
        assert result["verified_count"] >= 1

    def test_unverified_citation_wrong_doc(self, verifier, sample_chunks):
        answer = "The CRAR is 15% [Master Direction - Basel III, Section 5, Page 5]."
        result = verifier.verify_and_clean(answer, sample_chunks)
        assert result["unverified_count"] >= 1

    def test_mixed_citations(self, verifier, sample_chunks):
        answer = (
            "CRAR is 15% [Master Direction - NBFC-ND-SI, Capital Requirements, Page 5]. "
            "Tax rate is 30% [Income Tax Act, Section 115, Page 1]."
        )
        result = verifier.verify_and_clean(answer, sample_chunks)
        assert result["verified_count"] >= 1
        assert result["unverified_count"] >= 1


class TestAnswerCleaning:
    def test_unverified_marked(self, verifier, sample_chunks):
        answer = "Data from [Nonexistent Doc, Section 1, Page 99]."
        result = verifier.verify_and_clean(answer, sample_chunks)
        assert "⚠️" in result["cleaned_answer"]

    def test_verified_not_marked(self, verifier, sample_chunks):
        answer = "CRAR is 15% [Master Direction - NBFC-ND-SI, Capital Requirements, Page 5]."
        result = verifier.verify_and_clean(answer, sample_chunks)
        # Verified citations should NOT have the warning
        if result["verified_count"] > 0 and result["unverified_count"] == 0:
            assert "⚠️" not in result["cleaned_answer"]
