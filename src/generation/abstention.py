"""
Abstention Checker — decides whether the system should refuse to answer
based on retrieval confidence scores.

When the top reranked score falls below a calibrated threshold, the system
abstains rather than risking an unsupported answer.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Standard refusal message
ABSTENTION_MESSAGE = (
    "I cannot answer this question from the available policy documents. "
    "The retrieved context does not contain sufficient information to provide "
    "a reliable, well-cited answer. Please try rephrasing your question or "
    "consult the relevant RBI master direction directly."
)


class AbstentionChecker:
    """
    Decides whether to abstain from answering based on retrieval quality.

    Checks:
    1. **Score threshold**: Max reranked score must exceed a tuned threshold.
    2. **Minimum chunks**: At least N chunks must have non-trivial scores.
    3. **Score spread**: Top score shouldn't be an outlier (all others near-zero).

    The threshold is calibrated on the 120-question gold evaluation set to
    maximize abstention recall on unanswerable questions while minimizing
    false abstentions on answerable ones.
    """

    def __init__(
        self,
        threshold: float | None = None,
        min_supporting_chunks: int = 2,
        min_secondary_score_ratio: float = 0.3,
    ) -> None:
        settings = get_settings()
        self.threshold = threshold or settings.abstention_threshold
        self.min_supporting_chunks = min_supporting_chunks
        self.min_secondary_score_ratio = min_secondary_score_ratio

    def should_abstain(
        self,
        reranked_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Determine whether to abstain from answering.

        Args:
            reranked_results: Results from the cross-encoder reranker,
                            each with a 'rerank_score' field.

        Returns:
            Dict with:
                - 'abstain': bool
                - 'reason': str explaining the decision
                - 'max_score': float
                - 'num_supporting': int
                - 'threshold': float
        """
        if not reranked_results:
            return {
                "abstain": True,
                "reason": "No results retrieved",
                "max_score": 0.0,
                "num_supporting": 0,
                "threshold": self.threshold,
            }

        scores = [r.get("rerank_score", 0.0) for r in reranked_results]
        max_score = max(scores)
        num_supporting = sum(1 for s in scores if s > self.threshold * 0.5)

        # Check 1: Score threshold
        if max_score < self.threshold:
            reason = (
                f"Max rerank score ({max_score:.4f}) below threshold "
                f"({self.threshold:.4f})"
            )
            logger.info("Abstaining: %s", reason)
            return {
                "abstain": True,
                "reason": reason,
                "max_score": max_score,
                "num_supporting": num_supporting,
                "threshold": self.threshold,
            }

        # Check 2: Minimum supporting chunks
        if num_supporting < self.min_supporting_chunks:
            reason = (
                f"Only {num_supporting} supporting chunk(s), "
                f"need at least {self.min_supporting_chunks}"
            )
            logger.info("Abstaining: %s", reason)
            return {
                "abstain": True,
                "reason": reason,
                "max_score": max_score,
                "num_supporting": num_supporting,
                "threshold": self.threshold,
            }

        # Check 3: Score spread — if only the top result is relevant
        # and all others are near-zero, it might be a false positive
        if len(scores) >= 2:
            second_score = sorted(scores, reverse=True)[1]
            if second_score < max_score * self.min_secondary_score_ratio:
                # Only warn, don't abstain — single strong match is still useful
                logger.warning(
                    "Weak score spread: top=%.4f, second=%.4f (ratio=%.2f)",
                    max_score,
                    second_score,
                    second_score / max_score if max_score > 0 else 0,
                )

        logger.info(
            "Proceeding to answer: max_score=%.4f, supporting=%d, threshold=%.4f",
            max_score,
            num_supporting,
            self.threshold,
        )
        return {
            "abstain": False,
            "reason": "Sufficient retrieval confidence",
            "max_score": max_score,
            "num_supporting": num_supporting,
            "threshold": self.threshold,
        }

    @staticmethod
    def get_abstention_response() -> dict[str, Any]:
        """Return the standard abstention response."""
        return {
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
            "abstained": True,
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "latency_ms": 0,
        }
