"""
RAG Generator — produces cited answers from retrieved context using an LLM.

Enforces a strict citation format [Document Name, Section X.Y, Page Z]
and uses chain-of-thought prompting to reduce hallucination.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Credit Policy & Regulatory Assistant specializing in RBI (Reserve Bank of India) master directions, NBFC regulations, and credit policy circulars.

## Instructions

1. **Answer ONLY from the provided context.** Do not use any external knowledge.
2. **Every factual claim must have a citation** in the format: [Document Name, Section/Heading, Page N]
3. **If the context does not contain sufficient information to answer the question, say exactly:**
   "I cannot answer this question from the available policy documents. The retrieved context does not contain the required information."
4. **Do not guess or infer beyond what the context states.**
5. **For multi-part questions**, address each part with its own citations.
6. **Use precise regulatory language** from the source documents.

## Citation Format
- Always cite as: [Document Title, Section Heading, Page X]
- If section hierarchy is available, use the most specific heading.
- Example: [Master Direction - NBFC-ML, Chapter III > Capital Requirements, Page 12]

## Response Structure
1. First, identify the relevant passages from the context.
2. Then, synthesize a clear, accurate answer with inline citations.
3. If parts of the question cannot be answered from context, explicitly state which parts.

## Context
{context}

## Question
{question}
"""

# ── User Prompt (for providers that separate system/user) ────────────────
USER_PROMPT = """Based on the provided regulatory context, please answer the following question with precise citations.

Question: {question}

Remember:
- Cite every factual claim as [Document Name, Section, Page N]
- If you cannot find the answer in the context, say so explicitly
- Use the exact language from the regulatory documents where possible"""


class RAGGenerator:
    """
    LLM-powered answer generator with citation enforcement.

    Supports OpenAI and Google Generative AI backends.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the LLM client based on provider setting."""
        if self.settings.llm_provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.settings.openai_api_key)
                self._provider = "openai"
                logger.info("Using OpenAI provider with model: %s", self.settings.llm_model)
            except ImportError:
                logger.error("openai package not installed")
                raise
        elif self.settings.llm_provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.settings.google_api_key)
                self._client = genai.GenerativeModel(self.settings.llm_model)
                self._provider = "google"
                logger.info("Using Google provider with model: %s", self.settings.llm_model)
            except ImportError:
                logger.error("google-generativeai package not installed")
                raise
        else:
            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")

    def generate(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Generate a cited answer from retrieved context chunks.

        Args:
            question: The user's question.
            context_chunks: Reranked chunks from the retrieval pipeline.
            temperature: LLM sampling temperature (low for factual accuracy).
            max_tokens: Maximum output tokens.

        Returns:
            Dict with 'answer', 'citations', 'token_usage', 'latency_ms'.
        """
        t_start = time.perf_counter()

        # Build context string from chunks
        context_str = self._format_context(context_chunks)

        # Generate based on provider
        if self._provider == "openai":
            result = self._generate_openai(
                question, context_str, temperature, max_tokens
            )
        else:
            result = self._generate_google(
                question, context_str, temperature, max_tokens
            )

        result["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)

        logger.info(
            "Generated answer: %d chars, %d tokens, %.0fms",
            len(result.get("answer", "")),
            result.get("token_usage", {}).get("total", 0),
            result["latency_ms"],
        )
        return result

    def _format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a numbered context string."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            section = " > ".join(chunk.get("section_hierarchy", [])) or "N/A"
            header = (
                f"[Context {i}]\n"
                f"Document: {chunk.get('document_title', 'Unknown')}\n"
                f"Section: {section}\n"
                f"Page: {chunk.get('page_number', 'N/A')}\n"
                f"---\n"
            )
            parts.append(f"{header}{chunk.get('text', '')}\n")
        return "\n".join(parts)

    def _generate_openai(
        self,
        question: str,
        context_str: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Generate using OpenAI API."""
        system_msg = SYSTEM_PROMPT.format(context=context_str, question=question)

        response = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": USER_PROMPT.format(question=question)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage

        return {
            "answer": answer,
            "token_usage": {
                "prompt": usage.prompt_tokens if usage else 0,
                "completion": usage.completion_tokens if usage else 0,
                "total": usage.total_tokens if usage else 0,
            },
        }

    def _generate_google(
        self,
        question: str,
        context_str: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Generate using Google Generative AI."""
        prompt = SYSTEM_PROMPT.format(context=context_str, question=question)

        response = self._client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        answer = response.text or ""

        # Google API doesn't always expose token counts cleanly
        prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
        completion_tokens = len(answer.split()) * 1.3

        return {
            "answer": answer,
            "token_usage": {
                "prompt": int(prompt_tokens),
                "completion": int(completion_tokens),
                "total": int(prompt_tokens + completion_tokens),
            },
        }
