import json

from app.database.connection import SessionLocal
from app.rag.retrieval import retrieve
from app.tools.answer_evaluation import evaluate_answer
from app.tools.quiz_generation import generate_quiz
from app.services.assessment_service import save_assessment_result


def main():
    # 1. Retrieve real course content from RAG.
    query = "What is RAG?"

    results = retrieve(
        query=query,
        top_k=5,
    )

    if not results:
        raise ValueError("RAG returned no results.")

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # Build unique topics from the retrieved material.
    topic_map = {}

    for result in results:
        topic_id = result.get("topic_id")
        source = result.get("source", {})

        if topic_id is not None and topic_id not in topic_map:
            topic_map[topic_id] = {
                "topic_id": topic_id,
                "topic": source.get(
                    "lesson_name",
                    f"Topic {topic_id}",
                ),
            }

    topics = list(topic_map.values())

    if not topics:
        raise ValueError("No topics were found in RAG results.")

    print("RAG TOPICS:")
    for topic in topics:
        print(topic)

    # 2. Generate diagnostic quiz from real RAG context.
    quiz_result = generate_quiz.invoke(
        {
            "topics": topics,
            "context": context,
            "student_level": "beginner",
            "num_questions": 5,
            "assessment_type": "diagnostic",
        }
    )

    quiz = json.loads(quiz_result)

    print("\nQUIZ GENERATED:")
    print(json.dumps(quiz, ensure_ascii=False, indent=2))

    questions = quiz["questions"]

    # 3. Simulate learner answers and evaluate them.
    # First two answers are intentionally correct.
    # Remaining answers are intentionally incorrect.
    learner_answers = []

    for index, question in enumerate(questions):
        if index < 2:
            learner_answer = question["correct_answer"]
        else:
            learner_answer = next(
                option
                for option in question["options"]
                if option != question["correct_answer"]
            )

        evaluation_result = evaluate_answer.invoke(
            {
                "question": question["question_text"],
                "correct_answer": question["correct_answer"],
                "learner_answer": learner_answer,
            }
        )

        evaluation = json.loads(evaluation_result)

        learner_answers.append(
            {
                **question,
                "learner_answer": learner_answer,
                "is_correct": evaluation["is_correct"],
                "score_awarded": evaluation["score_awarded"],
                "feedback": evaluation["feedback"],
            }
        )

    print("\nEVALUATED ANSWERS:")
    print(
        json.dumps(
            learner_answers,
            ensure_ascii=False,
            indent=2,
        )
    )

    # 4. Save the complete assessment result.
    db = SessionLocal()

    try:
        attempt = save_assessment_result(
            db=db,
            user_id=1,
            topic_id=None,
            assessment_type="diagnostic",
            questions=learner_answers,
            feedback="Diagnostic assessment completed.",
        )

        print("\nASSESSMENT SAVED:")
        print("Attempt ID:", attempt.assessment_attempt_id)
        print("Score:", attempt.score)
        print("Max Score:", attempt.max_score)

    finally:
        db.close()


if __name__ == "__main__":
    main()
