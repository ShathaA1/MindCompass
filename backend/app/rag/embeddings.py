"""Generates OpenAI embeddings for chunks produced by rag/chunking.py.

Sends chunk text to OpenAI in batches (default 100 chunks/request) and
retries transient failures (rate limits, timeouts, server errors) with
exponential backoff, since a full knowledge-base ingestion run can make
hundreds of embedding calls.
"""

import logging
import time

import openai
from openai import OpenAI

from backend.app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_client: OpenAI | None = None

DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0


def _get_openai_client() -> OpenAI:
    """Lazily create a single shared OpenAI client."""

    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _embed_batch_with_retry(
    texts: list[str], model: str, max_retries: int = MAX_RETRIES
) -> list[list[float]]:
    """Embed one batch of texts, retrying transient errors with backoff."""

    client = _get_openai_client()
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(model=model, input=texts)
            # The API preserves input order, so zip-by-index is safe.
            return [item.embedding for item in response.data]
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            if attempt == max_retries:
                logger.error(
                    "Embedding batch failed after %d attempts: %s", attempt, exc
                )
                raise
            logger.warning(
                "Embedding batch failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt,
                max_retries,
                exc,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("unreachable")  # pragma: no cover


def embed_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of strings, batching requests and preserving order."""

    model = model or settings.openai_embedding_model
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(_embed_batch_with_retry(batch, model=model))

    return embeddings


def embed_chunks(
    chunks: list[dict],
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Attach an `embedding` field to each chunk dict from rag/chunking.py.

    Chunks with empty text are skipped (and excluded from the result)
    rather than sent to the API.
    """

    embeddable = [c for c in chunks if c.get("text", "").strip()]
    if not embeddable:
        return []

    texts = [c["text"] for c in embeddable]
    vectors = embed_texts(texts, model=model, batch_size=batch_size)

    enriched = []
    for chunk, vector in zip(embeddable, vectors, strict=True):
        new_chunk = dict(chunk)
        new_chunk["embedding"] = vector
        enriched.append(new_chunk)
    return enriched
