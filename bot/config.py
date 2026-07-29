"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _admin_ids() -> set[int]:
    raw = os.getenv("ADMIN_IDS", "")
    return {int(part) for part in raw.replace(" ", "").split(",") if part}


@dataclass(frozen=True)
class Config:
    bot_token: str
    anthropic_api_key: str
    database_url: str

    model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5")
    effort: str = os.getenv("CLAUDE_EFFORT", "low")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Retrieval
    top_k: int = _int_env("TOP_K", 5)
    # Cosine distance above which a chunk is considered irrelevant. pgvector's
    # `<=>` returns 1 - cosine_similarity, so 0.0 is identical and 1.0 unrelated.
    #
    # 0.40 was measured, not guessed: against the sample documents, on-topic
    # questions land at 0.19-0.33 and off-topic ones at 0.47-0.61, so anything
    # in (0.34, 0.47) separates them cleanly. Re-measure if you swap the
    # embedding model — the scale is model-specific.
    max_distance: float = float(os.getenv("MAX_DISTANCE", "0.40"))

    # Chunking, in characters. Small chunks keep citations precise and sharpen
    # the on/off-topic distance gap; the overlap stops a sentence that straddles
    # a boundary from being lost.
    chunk_size: int = _int_env("CHUNK_SIZE", 700)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 150)

    max_file_bytes: int = _int_env("MAX_FILE_MB", 20) * 1024 * 1024

    admin_ids: frozenset[int] = field(default_factory=lambda: frozenset(_admin_ids()))

    @classmethod
    def from_env(cls) -> "Config":
        missing = [
            name
            for name in ("BOT_TOKEN", "ANTHROPIC_API_KEY", "DATABASE_URL")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
        return cls(
            bot_token=os.environ["BOT_TOKEN"],
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            database_url=os.environ["DATABASE_URL"],
        )
