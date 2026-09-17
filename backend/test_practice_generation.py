import json

from app.rag.retrieval import retrieve
from app.tools.practice_generation import generate_practice


def main():
    query = "What is the attention mechanism in Transformers?"

    # Retrieve real course content from RAG
    results = retrieve(query=query, top_k=5)

    if not results:
        raise ValueError("RAG returned no results.")

    # Build learning context from retrieved course content
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # Get topic information from RAG metadata
    topic = None

    for result in results:
        source = result.get("source", {})
        lesson_name = source.get("lesson_name")

        if lesson_name:
            topic = lesson_name
            break

    if topic is None:
        topic = "Attention Mechanism in Transformers"

    weak_areas = """
[
    {
        "area": "Attention Mechanism",
        "reason": "The learner had difficulty identifying the purpose of the attention mechanism."
    }
]
"""

    print("RAG RESULTS:")
    for index, result in enumerate(results, start=1):
        source = result.get("source", {})

        print(f"\nResult {index}:")
        print(f"Similarity: {result.get('similarity_score')}")
        print(f"Topic ID: {result.get('topic_id')}")
        print(f"Source: {source}")

    print("\nPRACTICE TOPIC:")
    print(topic)

    result = generate_practice.invoke(
        {
            "topic": topic,
            "context": context,
            "student_level": "beginner",
            "practice_type": "true_false",
            "num_items": 5,
            "weak_areas": weak_areas,
        }
    )

    practice = json.loads(result)

    print("\nGENERATED PRACTICE:")
    print(json.dumps(practice, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()