"""Tests authentication, onboarding, and core FastAPI endpoints."""
from fastapi.testclient import TestClient

from app.main import app
from app.core.security import get_current_user
import app.api.learning as learning_api
import app.api.chat as chat_api
from app.database.connection import get_db

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
        assert data["topics_assessed"] == 0
        assert data["average_mastery"] is None
        assert data["topic_masteries"] == []
        assert data["weak_areas"] == []

    finally:
        # Always remove the authentication override
        # so it does not affect other API tests.
        app.dependency_overrides.clear()

def test_submit_chat_assessment(monkeypatch):
    """
    Test that the Tutor assessment endpoint:
    1. Authenticates the learner.
    2. Verifies the learner's chat session.
    3. Passes assessment questions and answers
       to the Tutor Agent.
    4. Returns mastery and recommendation data.
    """

    # ---------------------------------------------------------
    # Fake database
    # ---------------------------------------------------------

    class FakeSession:
        """Represent an existing learner chat session."""

        chat_session_id = 10
        user_id = 2

    class FakeQuery:
        """Provide the query operations used by the endpoint."""

        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return FakeSession()

    class FakeDB:
        """Prevent the test from accessing the real database."""

        def query(self, model):
            return FakeQuery()

    def fake_get_db():
        """Provide the fake database dependency."""

        yield FakeDB()

    # ---------------------------------------------------------
    # Fake Tutor Agent
    # ---------------------------------------------------------

    class FakeGraph:
        def invoke(self, state):
            # Verify authenticated learner and session.
            assert state["user_id"] == 2
            assert state["session_id"] == 10

            # Verify assessment metadata.
            assert state["assessment_type"] == "topic"

            # Verify questions are passed back to the graph.
            assert len(
                state["assessment_questions"]
            ) == 2

            # Verify learner answers are passed correctly.
            assert len(
                state["assessment_answers"]
            ) == 2

            assert (
                state["assessment_answers"][0][
                    "learner_answer"
                ]
                == "4"
            )

            assert (
                state["assessment_answers"][1][
                    "learner_answer"
                ]
                == "15"
            )

            # Simulate the result produced after
            # assessment evaluation and mastery update.
            return {
                "response": (
                    "Assessment completed. "
                    "Score: 2/2. "
                    "Mastery: 100%. "
                    "Recommended next step: recommend."
                ),
                "assessment_result": {
                    "assessment_attempt_id": 20,
                    "score": 2,
                    "max_score": 2,
                },
                "topic_mastery": {
                    "topic_id": 27,
                    "mastery_score": 100,
                    "weak_areas": [],
                },
                "recommended_action": "recommend",
                "recommendation_reason": (
                    "The learner has demonstrated "
                    "strong mastery of this topic."
                ),
            }

    # Prevent the API from running the real graph.
    monkeypatch.setattr(
        chat_api,
        "build_tutor_graph",
        lambda: FakeGraph(),
    )

    # Replace authentication and database access
    # with controlled test dependencies.
    app.dependency_overrides[get_current_user] = (
        fake_current_user
    )

    app.dependency_overrides[get_db] = (
        fake_get_db
    )

    try:
        response = client.post(
            "/chat/10/assessment",
            json={
                "assessment_type": "topic",
                "assessment_questions": [
                    {
                        "topic_id": 27,
                        "question_text": (
                            "What value is assigned "
                            "after a = 4?"
                        ),
                        "question_type": (
                            "multiple_choice"
                        ),
                        "difficulty": "beginner",
                        "options": [
                            "0",
                            "4",
                            "8",
                            "10",
                        ],
                        "correct_answer": "4",
                    },
                    {
                        "topic_id": 27,
                        "question_text": (
                            "What is the value after "
                            "a = 10; a = a + 5?"
                        ),
                        "question_type": (
                            "multiple_choice"
                        ),
                        "difficulty": "beginner",
                        "options": [
                            "5",
                            "10",
                            "15",
                            "50",
                        ],
                        "correct_answer": "15",
                    },
                ],
                "assessment_answers": [
                    {
                        "learner_answer": "4",
                    },
                    {
                        "learner_answer": "15",
                    },
                ],
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["session_id"] == 10

        assert (
            data["assessment_result"]["score"]
            == 2
        )

        assert (
            data["assessment_result"]["max_score"]
            == 2
        )

        assert (
            data["topic_mastery"]["mastery_score"]
            == 100
        )

        assert (
            data["recommended_action"]
            == "recommend"
        )

        assert (
            "Assessment completed"
            in data["response"]
        )

    finally:
        # Always remove dependency overrides so
        # other API tests are unaffected.
        app.dependency_overrides.clear()