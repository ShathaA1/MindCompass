from app.database.connection import SessionLocal
from app.services.learner_service import get_learner_context


def test_get_learner_context_user_not_found():
    db = SessionLocal()

    try:
        result = get_learner_context(
            db=db,
            user_id=999999
        )

        assert result == {}

    finally:
        db.close()

def test_get_learner_context_existing_user():
    db = SessionLocal()

    try:
        result = get_learner_context(
            db=db,
            user_id=2
        )

        assert result["user_id"] == 2
        assert result["name"] == "Test Learner"
        assert result["goal"] == "Learn Agentic AI"
        assert result["initial_level"] == "beginner"
        assert result["current_level"] == "beginner"
        assert result["weekly_hours"] == 5
        assert result["preferred_format"] == "practical_examples"
        assert result["preferred_pace"] == "moderate"

    finally:
        db.close()