from app.database.connection import SessionLocal
from app.agent.graph import (
    load_learner_context,
    load_active_learning_path,
)
from app.agent.planning import (
    determine_learner_need,
    route_action,
    plan_next_topic,
)

def test_load_learner_context():
    """
    Test that the agent node loads learner data
    from the database and returns it in learner_context.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Initial agent state
        state = {
            "user_id": 2,
            "session_id": 1,
            "user_message": "What should I learn next?"
        }

        # Run the learner context node
        result = load_learner_context(
            state=state,
            db=db
        )

        # Make sure learner_context was returned
        assert "learner_context" in result

        # Verify that the correct learner was loaded
        assert result["learner_context"]["user_id"] == 2
        assert result["learner_context"]["name"] == "Test Learner"

    finally:
        # Always close the database session after the test
        db.close()


def test_load_active_learning_path():
    """
    Test that the agent node loads the learner's
    active learning path from the database.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # The node only needs user_id to find the learning path
        state = {
            "user_id": 2,
            "session_id": 1,
            "user_message": "What should I learn next?"
        }

        # Run the learning path node
        result = load_active_learning_path(
            state=state,
            db=db
        )

        # Make sure learning_path was returned
        assert "learning_path" in result

        # Verify that the correct active path was loaded
        assert result["learning_path"]["learning_path_id"] == 1
        assert result["learning_path"]["user_id"] == 2
        assert result["learning_path"]["name"] == "Agentic AI Learning Path"
        assert result["learning_path"]["status"] == "active"

    finally:
        # Always close the database session after the test
        db.close()

def test_determine_learner_need_explain():
    """
    Test that explanation-related messages
    are classified as explain.
    """

    # Create a minimal TutorState for the test
    state = {
        "user_id": 2,
        "user_message": "Can you explain what an agent is?"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "explain"


def test_determine_learner_need_assess():
    """
    Test that quiz or assessment requests
    are classified as assess.
    """

    # Create a learner message asking for a quiz
    state = {
        "user_id": 2,
        "user_message": "Quiz me on this topic"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "assess"


def test_determine_learner_need_practice():
    """
    Test that practice-related requests
    are classified as practice.
    """

    # Create a learner message asking for practice
    state = {
        "user_id": 2,
        "user_message": "I want to practice this topic"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "practice"


def test_determine_learner_need_review():
    """
    Test that review-related requests
    are classified as review.
    """

    # Create a learner message asking for a review
    state = {
        "user_id": 2,
        "user_message": "Can we review the previous topic?"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "review"


def test_determine_learner_need_recommend():
    """
    Test that next-step requests
    are classified as recommend.
    """

    # Create a learner message asking what to learn next
    state = {
        "user_id": 2,
        "user_message": "What should I learn next?"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "recommend"


def test_determine_learner_need_default():
    """
    Test that unclear messages fall back
    to the default recommend action.
    """

    # Create a message that does not match any known intent
    state = {
        "user_id": 2,
        "user_message": "Hello"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the default behavior
    assert result["learner_need"] == "recommend"


def test_route_action_explain():
    """
    Test that explain requests are routed
    to the teaching node.
    """

    # Create a state with an explain decision
    state = {
        "learner_need": "explain"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "teach"


def test_route_action_assess():
    """
    Test that assessment requests are routed
    to the assessment node.
    """

    # Create a state with an assess decision
    state = {
        "learner_need": "assess"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "assess"


def test_route_action_practice():
    """
    Test that practice requests are routed
    to the practice node.
    """

    # Create a state with a practice decision
    state = {
        "learner_need": "practice"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "practice"


def test_route_action_review():
    """
    Test that review requests are routed
    to the review node.
    """

    # Create a state with a review decision
    state = {
        "learner_need": "review"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "review"


def test_route_action_recommend():
    """
    Test that recommendation requests are routed
    to the recommendation node.
    """

    # Create a state with a recommend decision
    state = {
        "learner_need": "recommend"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "recommend"


def test_route_action_default():
    """
    Test that an unknown learner need falls back
    to the recommendation node.
    """

    # Create a state with an unexpected value
    state = {
        "learner_need": "unknown"
    }

    # Route the action
    result = route_action(state)

    # Verify the safe default destination
    assert result == "recommend"


def test_plan_next_topic():
    """
    Test that the agent selects the next available topic
    from the learner's active learning path.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Simulate the TutorState after the active
        # learning path has already been loaded.
        state = {
            "user_id": 2,
            "learning_path": {
                "learning_path_id": 1,
                "name": "Agentic AI Learning Path",
                "status": "active",
            },
        }

        # Run the planning function
        result = plan_next_topic(
            state=state,
            db=db,
        )

        # Make sure current_topic was added
        assert "current_topic" in result

        # Python and Machine Learning are completed,
        # so Agentic AI should be selected next.
        assert result["current_topic"]["topic_id"] == 3
        assert result["current_topic"]["name"] == "Introduction to Agentic AI"

        # Verify its position and current status
        assert result["current_topic"]["position"] == 3
        assert result["current_topic"]["status"] == "pending"

    finally:
        # Always close the database session after the test
        db.close()