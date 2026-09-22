"""Tests authentication, onboarding, and core FastAPI endpoints."""
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import get_current_user
import app.api.learning as learning_api


client = TestClient(app)


def fake_current_user():
    """
    Return a fake authenticated learner for API testing.
    """

    return {
        "sub": "2",
        "email": "learner@example.com",
    }


def test_start_initial_diagnostic(monkeypatch):
    """
    Test that the diagnostic start endpoint:
    1. Gets the learner ID from authentication.
    2. Passes the selected path to the Tutor Agent.
    3. Returns the generated diagnostic data.
    """

    class FakeGraph:
        def invoke(self, state):
            # Verify that the API builds the correct TutorState.
            assert state["user_id"] == 2
            assert state["selected_path"] == "agentic_ai"

            return {
                "selected_path": "agentic_ai",
                "assessment_type": "diagnostic",
                "diagnostic_topics": [
                    {
                        "topic_id": 1,
                        "name": "Python Basics",
                    },
                    {
                        "topic_id": 2,
                        "name": "Machine Learning Basics",
                    },
                ],
                "assessment_questions": [
                    {
                        "topic_id": 1,
                        "question_text": "What is a Python variable?",
                        "question_type": "multiple_choice",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                    }
                ],
                "response": "Initial diagnostic generated.",
            }

    # Prevent the API test from running the real LangGraph,
    # RAG pipeline, or LLM.
    monkeypatch.setattr(
        learning_api,
        "build_tutor_graph",
        lambda: FakeGraph(),
    )

    # Replace JWT authentication with a controlled test user.
    app.dependency_overrides[get_current_user] = (
        fake_current_user
    )

    try:
        response = client.post(
            "/learning/diagnostic/start",
            json={
                "selected_path": "agentic_ai",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["selected_path"] == "agentic_ai"
        assert data["assessment_type"] == "diagnostic"
        assert len(data["diagnostic_topics"]) == 2
        assert len(data["assessment_questions"]) == 1
        assert data["message"] == (
            "Initial diagnostic generated."
        )

    finally:
        # Always remove the authentication override
        # so it does not affect other API tests.
        app.dependency_overrides.clear()


def test_submit_initial_diagnostic(monkeypatch):
    """
    Test that the diagnostic submit endpoint:
    1. Gets the learner ID from authentication.
    2. Passes the diagnostic questions and answers to the Tutor Agent.
    3. Returns the assessment result and personalized learning path.
    """

    class FakeGraph:
        def invoke(self, state):
            # Verify that the API builds the correct TutorState.
            assert state["user_id"] == 2
            assert state["selected_path"] == "agentic_ai"

            assert len(state["assessment_questions"]) == 2
            assert len(state["assessment_answers"]) == 2

            assert (
                state["assessment_answers"][0]["learner_answer"]
                == "A"
            )
            assert (
                state["assessment_answers"][1]["learner_answer"]
                == "B"
            )

            return {
                "selected_path": "agentic_ai",
                "assessment_result": {
                    "assessment_attempt_id": 10,
                    "score": 2,
                    "max_score": 2,
                },
                "learning_path": {
                    "learning_path_id": 5,
                    "name": "Agentic AI Learning Path",
                    "status": "active",
                },
                "current_topic": {
                    "topic_id": 3,
                    "name": "Introduction to Agentic AI",
                    "status": "pending",
                    "recommended_action": "explain",
                },
                "next_topic_id": 3,
                "response": (
                    "Diagnostic completed and learning path created."
                ),
            }

    # Prevent the API test from running the real LangGraph,
    # RAG pipeline, or LLM.
    monkeypatch.setattr(
        learning_api,
        "build_tutor_graph",
        lambda: FakeGraph(),
    )

    # Replace JWT authentication with a controlled test user.
    app.dependency_overrides[get_current_user] = (
        fake_current_user
    )

    try:
        response = client.post(
            "/learning/diagnostic/submit",
            json={
                "selected_path": "agentic_ai",
                "assessment_questions": [
                    {
                        "topic_id": 1,
                        "question_text": "What is a Python variable?",
                        "question_type": "multiple_choice",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "A",
                    },
                    {
                        "topic_id": 2,
                        "question_text": (
                            "What is supervised learning?"
                        ),
                        "question_type": "multiple_choice",
                        "options": ["A", "B", "C", "D"],
                        "correct_answer": "B",
                    },
                ],
                "assessment_answers": [
                    {
                        "learner_answer": "A",
                    },
                    {
                        "learner_answer": "B",
                    },
                ],
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["selected_path"] == "agentic_ai"
        assert data["assessment_result"]["score"] == 2

        assert (
            data["learning_path"]["name"]
            == "Agentic AI Learning Path"
        )

        assert data["current_topic"]["topic_id"] == 3
        assert data["next_topic_id"] == 3

        assert data["message"] == (
            "Diagnostic completed and learning path created."
        )

    finally:
        # Always remove the authentication override
        # so it does not affect other API tests.
        app.dependency_overrides.clear()

def test_dashboard_summary():
    """
    Test that the dashboard endpoint returns:
    1. The learner's active learning path.
    2. Learning path progress.
    3. The current topic and recommended action.
    4. Mastery and weak-area information.
    """

    # Replace JWT authentication with a controlled test user.
    app.dependency_overrides[get_current_user] = (
        fake_current_user
    )

    try:
        response = client.get(
            "/learning/dashboard"
        )

        assert response.status_code == 200

        data = response.json()

        # Verify the learner profile summary.
        assert "goal" in data
        assert "current_level" in data
        assert "weekly_hours" in data

        # Verify the personalized learning path.
        assert data["learning_path"] is not None
        assert (
            data["learning_path"]["name"]
            == "Agentic AI Learning Path"
        )
        assert (
            data["learning_path"]["status"]
            == "active"
        )

        # Verify learning path progress.
        assert data["total_topics"] == 3
        assert data["completed_topics"] == 2
        assert data["progress_percentage"] == 66.7

        # Verify the current topic and next action.
        assert data["current_topic"] is not None
        assert (
            data["current_topic"]["topic_name"]
            == "Introduction to Agentic AI"
        )
        assert data["recommended_action"] == "explain"

        # Verify mastery information is available.
        assert data["topics_assessed"] >= 1
        assert data["average_mastery"] is not None
        assert isinstance(
            data["topic_masteries"],
            list,
        )
        assert isinstance(
            data["weak_areas"],
            list,
        )

    finally:
        # Always remove the authentication override
        # so it does not affect other API tests.
        app.dependency_overrides.clear()