from app.rag.retrieval import retrieve
from app.tools.explain import explain


def run_explanation(question: str, student_level: str, context: str):
    return explain.invoke(
        {
            "topic": question,
            "student_level": student_level,
            "context": context,
        }
    )


def main():
    question = "What is RAG?"

    # Retrieve the same course context once.
    results = retrieve(
        query=question,
        top_k=5,
    )

    # Build context from retrieved results.
    context_parts = []

    for result in results:
        source = result.get("source", {})

        context_parts.append(
            f"Source: {source.get('document_name', 'Unknown')}\n"
            f"{result['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Test beginner.
    beginner_result = run_explanation(
        question,
        "beginner",
        context,
    )

    print("\n" + "=" * 70)
    print("BEGINNER")
    print("=" * 70)
    print(beginner_result)

    # Test advanced.
    advanced_result = run_explanation(
        question,
        "advanced",
        context,
    )

    print("\n" + "=" * 70)
    print("ADVANCED")
    print("=" * 70)
    print(advanced_result)

    # Show retrieved sources.
    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, result in enumerate(results, start=1):
        source = result.get("source", {})

        print(
            f"[{i}] "
            f"{source.get('document_name', 'Unknown')} "
            f"(similarity={result['similarity_score']:.4f})"
        )


if __name__ == "__main__":
    main()