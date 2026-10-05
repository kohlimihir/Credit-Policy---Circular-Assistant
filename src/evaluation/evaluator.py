"""
Evaluation Harness — computes retrieval and generation quality metrics.

Metrics:
- Recall@K and MRR for retrieval
- Answer faithfulness and context precision (via LLM judge)
- Abstention precision / recall
- p50 / p95 latency
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates the RAG pipeline against a gold evaluation set.

    The gold set is a JSON file with entries like:
    {
        "question": "...",
        "expected_answer": "...",
        "relevant_chunks": ["chunk_id_1", ...],
        "relevant_documents": ["doc_name_1", ...],
        "category": "easy_lookup|multi_hop|inference|unanswerable",
        "answerable": true/false
    }
    """

    def __init__(self, gold_set_path: str | Path | None = None) -> None:
        self.gold_set: list[dict[str, Any]] = []
        if gold_set_path:
            self.load_gold_set(gold_set_path)

    def load_gold_set(self, path: str | Path) -> None:
        """Load the gold evaluation set from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            self.gold_set = json.load(f)
        logger.info("Loaded gold set: %d questions", len(self.gold_set))

    # ── Retrieval Metrics ────────────────────────────────────────────────

    # Retrieval is scored at DOCUMENT level, not chunk level.
    #
    # Chunk-level scoring is not possible with this gold set: chunk IDs are
    # UUIDs regenerated on every ingestion run, so they cannot be labelled in
    # advance. The gold set instead labels `relevant_documents` (source PDF
    # filenames), which are stable. Each retrieved chunk is therefore mapped to
    # its source document, and the ranked list of documents is scored.
    #
    # Consequence to be aware of: document-level Recall@5 is a weaker (more
    # forgiving) signal than chunk-level, because retrieving any chunk from the
    # right document counts as a hit even if it is the wrong clause.

    @staticmethod
    def recall_at_k(
        retrieved_docs: list[str],
        relevant_docs: list[str],
        k: int = 5,
    ) -> float:
        """
        Compute Recall@K: fraction of relevant documents surfaced in the top-K
        retrieved chunks.

        Args:
            retrieved_docs: Source document of each retrieved chunk, in rank
                            order. May contain duplicates (several chunks from
                            the same document).
            relevant_docs: Ground-truth relevant document names.
            k: Cut-off rank, applied to the chunk list.

        Returns:
            Recall@K score (0.0 to 1.0).
        """
        if not relevant_docs:
            return 1.0  # nothing to find, treated as satisfied

        top_k_docs = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        found = len(top_k_docs & relevant_set)
        return found / len(relevant_set)

    @staticmethod
    def mrr(
        retrieved_docs: list[str],
        relevant_docs: list[str],
    ) -> float:
        """
        Compute Reciprocal Rank: 1/rank of the first relevant document.

        Args:
            retrieved_docs: Source document of each retrieved chunk, in rank order.
            relevant_docs: Ground-truth relevant document names.

        Returns:
            Reciprocal rank (0.0 to 1.0). Averaged across queries this is MRR.
        """
        relevant_set = set(relevant_docs)
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                return 1.0 / rank
        return 0.0

    # ── Abstention Metrics ───────────────────────────────────────────────

    @staticmethod
    def abstention_metrics(
        predictions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Compute abstention precision and recall.

        Args:
            predictions: List of dicts with:
                - 'abstained': bool (system's decision)
                - 'answerable': bool (ground truth)

        Returns:
            Dict with 'precision', 'recall', 'f1', 'accuracy'.
        """
        # For abstention, a "positive" is correctly abstaining on an unanswerable question
        tp = sum(1 for p in predictions if p["abstained"] and not p["answerable"])
        fp = sum(1 for p in predictions if p["abstained"] and p["answerable"])
        fn = sum(1 for p in predictions if not p["abstained"] and not p["answerable"])
        tn = sum(1 for p in predictions if not p["abstained"] and p["answerable"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(predictions) if predictions else 0.0

        return {
            "abstention_precision": round(precision, 4),
            "abstention_recall": round(recall, 4),
            "abstention_f1": round(f1, 4),
            "abstention_accuracy": round(accuracy, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }

    # ── Latency Metrics ─────────────────────────────────────────────────

    @staticmethod
    def latency_metrics(latencies_ms: list[float]) -> dict[str, float]:
        """
        Compute latency percentiles.

        Args:
            latencies_ms: List of latency values in milliseconds.

        Returns:
            Dict with 'p50', 'p75', 'p90', 'p95', 'p99', 'mean', 'max'.
        """
        if not latencies_ms:
            return {k: 0.0 for k in ["p50", "p75", "p90", "p95", "p99", "mean", "max"]}

        arr = np.array(latencies_ms)
        return {
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "p99": round(float(np.percentile(arr, 99)), 1),
            "mean": round(float(np.mean(arr)), 1),
            "max": round(float(np.max(arr)), 1),
        }

    # ── LLM-Judge Faithfulness (simplified) ──────────────────────────────

    @staticmethod
    def faithfulness_score(
        answer: str,
        context_chunks: list[dict[str, Any]],
    ) -> float:
        """
        Simple heuristic faithfulness score.

        Checks what fraction of sentences in the answer contain information
        that can be traced back to the context. For a full LLM-judge
        evaluation, use the Ragas integration in benchmark.py.

        Returns:
            Score between 0.0 and 1.0.
        """
        if not answer.strip():
            return 0.0

        context_text = " ".join(c.get("text", "") for c in context_chunks).lower()
        context_words = set(context_text.split())

        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
        if not sentences:
            return 1.0

        grounded_count = 0
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            # A sentence is "grounded" if > 50% of its content words appear in context
            content_words = {w for w in sentence_words if len(w) > 3}
            if not content_words:
                grounded_count += 1
                continue
            overlap = len(content_words & context_words) / len(content_words)
            if overlap > 0.5:
                grounded_count += 1

        return grounded_count / len(sentences)

    # ── Aggregate Evaluation ─────────────────────────────────────────────

    def compute_aggregate_metrics(
        self,
        results: list[dict[str, Any]],
        evaluate_abstention: bool = True,
    ) -> dict[str, Any]:
        """
        Compute all metrics from a list of evaluation results.

        Args:
            results: List of dicts, each with:
                - 'retrieved_documents': list[str]  source doc per retrieved chunk, ranked
                - 'relevant_documents': list[str]   ground-truth doc names
                - 'answerable': bool
                - 'abstained': bool
                - 'answer': str
                - 'context_chunks': list[dict]
                - 'latency_ms': float
            evaluate_abstention: Set False for configurations where the
                abstention score is not on a comparable scale (e.g. runs
                without reranking), so the metric is reported as unevaluated
                rather than as a misleading number.

        Returns:
            Comprehensive metrics dict.
        """
        # Retrieval metrics, scored only on questions that have labelled
        # documents. The 20 unanswerable questions have none, and are measured
        # by the abstention metrics instead.
        scorable = [r for r in results if r.get("relevant_documents")]

        def mean_recall(k: int) -> float:
            if not scorable:
                return 0.0
            return round(
                float(
                    np.mean([
                        self.recall_at_k(r["retrieved_documents"], r["relevant_documents"], k=k)
                        for r in scorable
                    ])
                ),
                4,
            )

        mrr_scores = [
            self.mrr(r["retrieved_documents"], r["relevant_documents"]) for r in scorable
        ]

        # Per-category breakdown: easy_lookup should score far higher than
        # multi_hop, which needs several documents inside one 5-chunk window.
        by_category: dict[str, Any] = {}
        for category in sorted({r.get("category", "unknown") for r in scorable}):
            subset = [r for r in scorable if r.get("category") == category]
            by_category[category] = {
                "n": len(subset),
                "recall_at_5": round(
                    float(np.mean([
                        self.recall_at_k(r["retrieved_documents"], r["relevant_documents"], k=5)
                        for r in subset
                    ])), 4,
                ),
                "mrr": round(
                    float(np.mean([
                        self.mrr(r["retrieved_documents"], r["relevant_documents"])
                        for r in subset
                    ])), 4,
                ),
            }

        # Abstention metrics
        if evaluate_abstention:
            abstention = self.abstention_metrics(results)
        else:
            abstention = {"evaluated": False, "reason": "no calibrated score for this configuration"}

        # Faithfulness metrics
        faithfulness_scores = [
            self.faithfulness_score(r["answer"], r["context_chunks"])
            for r in results
            if not r.get("abstained") and r.get("answer")
        ]

        # Latency metrics
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        latency = self.latency_metrics(latencies)

        return {
            "retrieval": {
                "recall_at_1": mean_recall(1),
                "recall_at_3": mean_recall(3),
                "recall_at_5": mean_recall(5),
                "mrr": round(float(np.mean(mrr_scores)), 4) if mrr_scores else 0.0,
                "n_evaluated": len(scorable),
                "level": "document",
                "by_category": by_category,
            },
            "abstention": abstention,
            "faithfulness": {
                "mean": round(float(np.mean(faithfulness_scores)), 4)
                if faithfulness_scores
                else 0.0,
                "min": round(float(np.min(faithfulness_scores)), 4)
                if faithfulness_scores
                else 0.0,
                "n_evaluated": len(faithfulness_scores),
            },
            "latency": latency,
            "total_questions": len(results),
        }
