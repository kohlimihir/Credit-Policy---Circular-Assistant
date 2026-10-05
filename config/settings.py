"""
Credit Policy & Circular Assistant — Configuration
Pydantic Settings for type-safe, env-backed configuration.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root (two levels up from this file) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    google_api_key: str = ""
    # "none" = retrieval-only mode: no LLM call, answer is built from top passages
    llm_provider: Literal["openai", "google", "none"] = "openai"
    llm_model: str = "gpt-4o"

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "credit_policy"

    # ── Paths ────────────────────────────────────────────────────────────
    pdf_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "raw_pdfs")
    processed_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")

    # ── Retrieval ────────────────────────────────────────────────────────
    dense_top_k: int = 50
    sparse_top_k: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 5
    abstention_threshold: float = 0.35

    # ── Chunking Defaults ────────────────────────────────────────────────
    fixed_chunk_size: int = 512          # tokens
    recursive_chunk_size: int = 512      # tokens
    recursive_chunk_overlap: int = 128   # tokens
    structural_max_chunk_size: int = 1024  # tokens

    # ── Server ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


# ── Singleton ────────────────────────────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
