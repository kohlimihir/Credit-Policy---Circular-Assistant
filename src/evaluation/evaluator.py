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

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int = 5,
    ) -> float:
        """
        Compute Recall@K: fraction of relevant items found in top-K results.

        Args:
            retrieved_ids: Ordered list of retrieved chunk IDs.
            relevant_ids: Ground-truth relevant chunk IDs.
            k: Cut-off rank.

        Returns:
            Recall@K score (0.0 to 1.0).
        """
        if not relevant_ids:
            return 1.0  # no relevant items to find

        top_k_retrieved = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        found = len(top_k_retrieved & relevant_set)
        return found / len(relevant_set)

    @staticmethod
    def mrr(
        retrieved_ids: list[str],
        relevant_ids: list[str],
    ) -> float:
        """
        Compute Mean Reciprocal Rank: 1/rank of the first relevant result.

        Args:
            retrieved_ids: Ordered list of retrieved chunk IDs.
            relevant_ids: Ground-truth relevant chunk IDs.

        Returns:
            MRR score (0.0 to 1.0).
        """
        relevant_set = set(relevant_ids)
        for rank, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_set:
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
    ) -> dict[str, Any]:
        """
        Compute all metrics from a list of evaluation results.

        Args:
            results: List of dicts, each with:
                - 'retrieved_ids': list[str]
                - 'relevant_ids': list[str]
                - 'answerable': bool
                - 'abstained': bool
                - 'answer': str
                - 'context_chunks': list[dict]
                - 'latency_ms': float

        Returns:
            Comprehensive metrics dict.
        """
        # Retrieval metrics
        recall_scores = [
            self.recall_at_k(r["retrieved_ids"], r["relevant_ids"], k=5)
            for r in results
            if r.get("relevant_ids")
        ]
        mrr_scores = [
            self.mrr(r["retrieved_ids"], r["relevant_ids"])
            for r in results
            if r.get("relevant_ids")
        ]

        # Abstention metrics
        abstention = self.abstention_metrics(results)

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
                "recall_at_5": round(float(np.mean(recall_scores)), 4)
                if recall_scores
                else 0.0,
                "mrr": round(float(np.mean(mrr_scores)), 4)
                if mrr_scores
                else 0.0,
                "n_evaluated": len(recall_scores),
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
