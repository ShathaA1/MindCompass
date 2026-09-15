"""Stores embedded chunks in ChromaDB with per-chunk metadata.

Chunk IDs are derived deterministically from (source_path, chunk_index)
so that re-running ingestion on an unchanged file upserts the exact
same IDs instead of creating duplicates. Skipping already-processed
FILES entirely (via `learning_materials.processing_status` in
PostgreSQL) is the ingestion orchestrator's job, not this module's --
this module only knows how to write/query chunks that are handed to it.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from backend.app.core.config import get_settings

settings = get_settings()

_client: chromadb.ClientAPI | None = None

# Maps file_type -> a coarse "source_type" label used for filtering/UX.
SOURCE_TYPE_BY_FILE_TYPE = {
    "pdf": "slide",
    "ipynb": "lab",
    "html": "reference",
}


def _get_client() -> chromadb.ClientAPI:
    """Lazily create a single shared persistent ChromaDB client."""

    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection() -> Collection:
    """Return (creating if needed) the project's ChromaDB collection."""

    client = _get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def make_chunk_id(source_path: str, chunk_index: int) -> str:
    """Build a stable, reproducible chunk id from its source and position.

    The same (source_path, chunk_index) pair always yields the same id,
    so re-ingesting an unchanged file overwrites the same ChromaDB
    entries instead of creating duplicates.
    """

    raw = f"{source_path}::{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_chunk_metadata(
    chunk: dict,
    topic_id: int,
    course_name: str,
    chapter_number: int,
    lesson_name: str,
) -> dict:
    """Combine a chunk's own fields with topic-level context into one
    ChromaDB-safe metadata dict (str/int/float/bool values only).
    """

    file_type = chunk["file_type"]
    return {
        "topic_id": topic_id,
        "course_name": course_name,
        "chapter_number": chapter_number,
        "lesson_name": lesson_name,
        "document_name": Path(chunk["source_path"]).name,
        "source_path": chunk["source_path"],
        "file_type": file_type,
        "source_type": SOURCE_TYPE_BY_FILE_TYPE.get(file_type, "other"),
        "unit_type": chunk["unit_type"],
        "start_unit_index": chunk["start_unit_index"],
        "end_unit_index": chunk["end_unit_index"],
        "chunk_index": chunk["chunk_index"],
    }


def upsert_chunks(
    chunks: list[dict],
    topic_id: int,
    course_name: str,
    chapter_number: int,
    lesson_name: str,
) -> int:
    """Upsert a list of embedded chunks (from rag/embeddings.py) into Chroma.

    Each chunk dict must already contain an "embedding" field. Returns
    the number of chunks written.
    """

    if not chunks:
        return 0

    collection = get_collection()

    ids = [make_chunk_id(c["source_path"], c["chunk_index"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [
        build_chunk_metadata(c, topic_id, course_name, chapter_number, lesson_name)
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)
