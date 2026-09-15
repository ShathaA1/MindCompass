"""Retrieves relevant chunks from ChromaDB, filtered by topic_id, with sources.

Supports filtering by a single topic_id, a list of topic_ids (e.g. every
lesson in a chapter), or no filter at all (search the whole knowledge
base). Every result includes a `source` block so the caller can show
the learner exactly which document/pages/cells an answer came from.
"""

from backend.app.rag.embeddings import embed_texts
from backend.app.rag.vector_store import get_collection


def _build_topic_filter(topic_ids: int | list[int] | None) -> dict | None:
    """Turn a single id / list of ids / None into a ChromaDB `where` filter."""

    if topic_ids is None:
        return None
    if isinstance(topic_ids, int):
        return {"topic_id": topic_ids}
    if isinstance(topic_ids, list):
        if not topic_ids:
            return None
        if len(topic_ids) == 1:
            return {"topic_id": topic_ids[0]}
        return {"topic_id": {"$in": topic_ids}}
    raise TypeError("topic_ids must be an int, a list of ints, or None")


def _format_result(document: str, metadata: dict, distance: float) -> dict:
    """Shape one raw ChromaDB match into a result with a clear `source`."""

    return {
        "text": document,
        "similarity_score": 1 - distance,  # cosine distance -> similarity
        "topic_id": metadata.get("topic_id"),
        "source": {
            "document_name": metadata.get("document_name"),
            "course_name": metadata.get("course_name"),
            "chapter_number": metadata.get("chapter_number"),
            "lesson_name": metadata.get("lesson_name"),
            "source_type": metadata.get("source_type"),
            "file_type": metadata.get("file_type"),
            "start_unit_index": metadata.get("start_unit_index"),
            "end_unit_index": metadata.get("end_unit_index"),
        },
    }


def retrieve(
    query: str,
    topic_ids: int | list[int] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Semantic search over the knowledge base, optionally scoped to topics.

    Args:
        query: The learner's natural-language question.
        topic_ids: A single topic_id, a list of topic_ids, or None to
            search across every topic.
        top_k: Maximum number of chunks to return.

    Returns:
        A list of result dicts (highest similarity first), each with the
        chunk text, a similarity score, and a `source` block for citing
        where the answer came from.
    """

    collection = get_collection()
    query_embedding = embed_texts([query])[0]
    where_filter = _build_topic_filter(topic_ids)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    raw = collection.query(**query_kwargs)

    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    return [
        _format_result(doc, meta, dist)
        for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
    ]
