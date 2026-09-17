from app.database.connection import SessionLocal
from app.services.assessment_service import save_assessment_result


questions = [
    {
        "topic_id": 1,
        "question_text": "What is Python?",
        "question_type": "multiple_choice",
        "options": [
            "A programming language",
            "A database",
            "A machine learning model",
            "An operating system",
        ],
        "learner_answer": "A programming language",
        "correct_answer": "A programming language",
        "is_correct": True,
        "score_awarded": 1,
        "feedback": "Correct.",
    },
    {
        "topic_id": 1,
        "question_text": "What are variables used for in Python?",
        "question_type": "multiple_choice",
        "options": [
            "To store values",
            "To train models",
            "To label data",
            "To create databases",
        ],
        "learner_answer": "To store values",
        "correct_answer": "To store values",
        "is_correct": True,
        "score_awarded": 1,
        "feedback": "Correct.",
    },
    {
        "topic_id": 2,
        "question_text": "What type of data does supervised learning use?",
        "question_type": "multiple_choice",
        "options": [
            "Labeled data",
            "Only images",
            "No data",
            "Random data",
        ],
        "learner_answer": "Random data",
        "correct_answer": "Labeled data",
        "is_correct": False,
        "score_awarded": 0,
        "feedback": "Incorrect. Supervised learning uses labeled data.",
    },
]


db = SessionLocal()

try:
    attempt = save_assessment_result(
        db=db,
        user_id=1,
        topic_id=None,
        assessment_type="diagnostic",
        questions=questions,
        feedback="Diagnostic assessment completed.",
    )

    print("Assessment saved successfully.")
    print("Attempt ID:", attempt.assessment_attempt_id)
    print("Score:", attempt.score)
    print("Max Score:", attempt.max_score)

finally:
    db.close()
