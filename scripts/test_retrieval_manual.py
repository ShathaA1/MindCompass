"""
Quick manual script to test RAG retrieval quality for MindCompass.

Unlike tests/test_rag.py (automated pytest suite), this is an interactive
tool for manually trying real questions against the live ChromaDB and
inspecting similarity scores + source attribution.

Usage (from the project root, with venv activated):
    python -m scripts.test_retrieval_manual "What is prompt engineering?"
    python -m scripts.test_retrieval_manual "What is prompt engineering?" --topic 3
    python -m scripts.test_retrieval_manual "What is prompt engineering?" --top-k 10
"""

import argparse
import sys

from backend.app.rag.retrieval import retrieve


def main():
    parser = argparse.ArgumentParser(description="Test MindCompass RAG retrieval")
    parser.add_argument("query", type=str, help="The question to search for")
    parser.add_argument(
        "--topic", type=int, default=None, help="Optional topic_id to filter by"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return (default 5)"
    )
    args = parser.parse_args()

    print(f"\nQuery: {args.query!r}")
    print(f"Topic filter: {args.topic if args.topic is not None else 'None (search all topics)'}")
    print(f"Top K: {args.top_k}\n")
    print("-" * 80)

    try:
        results = retrieve(args.query, topic_ids=args.topic, top_k=args.top_k)
    except Exception as e:
        print(f"ERROR during retrieval: {e}")
        sys.exit(1)

    if not results:
        print("No results found. Check that ChromaDB has data and the topic_id (if used) exists.")
        return

    for i, r in enumerate(results, start=1):
        src = r["source"]
        print(f"\n[{i}] similarity_score = {r['similarity_score']:.4f}  (topic_id={r['topic_id']})")
        print(f"    Course:   {src.get('course_name')}")
        print(f"    Chapter:  {src.get('chapter_number')}  |  Lesson: {src.get('lesson_name')}")
        print(f"    File:     {src.get('document_name')}  ({src.get('file_type')})")
        print(f"    Snippet:  {r['text'][:200].strip()}...")

    print("\n" + "-" * 80)
    print(f"Returned {len(results)} result(s).\n")


if __name__ == "__main__":
    main()
