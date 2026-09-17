from app.rag.retrieval import retrieve
from app.tools.quiz_generation import generate_quiz


query = "What is RAG?"

results = retrieve(query, top_k=5)

context = "\n\n".join(
    result["text"]
    for result in results
)

topic_map = {}

for result in results:
    topic_id = result.get("topic_id")
    source = result.get("source", {})

    if topic_id is not None and topic_id not in topic_map:
        topic_map[topic_id] = {
            "topic_id": topic_id,
            "topic": source.get("lesson_name", f"Topic {topic_id}"),
        }

topics = list(topic_map.values())

if not topics:
    raise ValueError("No topics were found in RAG results.")

print("TOPICS:")
for topic in topics:
    print(topic)

print("\nGENERATING QUIZ...\n")

result = generate_quiz.invoke({
    "topics": topics,
    "context": context,
    "student_level": "beginner",
    "num_questions": 5,
    "assessment_type": "diagnostic",
})

print(result)
