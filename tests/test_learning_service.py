from app.database.connection import SessionLocal
from app.services.learning_service import (
    get_active_learning_path,
    select_next_topic,
    are_prerequisites_completed,
)

def test_get_active_learning_path():
    """
    Test that the learner's active learning path
    is loaded correctly from the database.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Load the active learning path for the test learner
        result = get_active_learning_path(
            db=db,
            user_id=2
        )

        # Make sure a learning path was returned
        assert result != {}

        # Check that the correct learning path was loaded
        assert result["learning_path_id"] == 1
        assert result["user_id"] == 2
        assert result["name"] == "Agentic AI Learning Path"
        assert result["goal"] == "Learn Agentic AI"
        assert result["status"] == "active"

    finally:
        # Always close the database session after the test
        db.close()


def test_select_next_topic():
    """
    Test that Agentic AI is selected after
    all of its prerequisites are completed.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Select the next available topic from learning path 1
        result = select_next_topic(
            db=db,
            learning_path_id=1
        )

        # Make sure a topic was returned
        assert result != {}

        # Python and ML are completed,
        # so Agentic AI should now be selected
        assert result["topic_id"] == 3
        assert result["name"] == "Introduction to Agentic AI"

        # Verify its position in the learning path
        assert result["position"] == 3

        # The selected topic should still be incomplete
        assert result["status"] == "pending"

    finally:
        # Always close the database session after the test
        db.close()

def test_prerequisites_completed_for_machine_learning():
    """
    Test that Machine Learning Basics is available
    when Python Basics has already been completed.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Topic 2 requires Topic 1.
        # Topic 1 is already completed in learning path 1.
        result = are_prerequisites_completed(
            db=db,
            learning_path_id=1,
            topic_id=2
        )

        # All prerequisites should be completed
        assert result is True

    finally:
        # Always close the database session after the test
        db.close()


def test_prerequisites_completed_for_agentic_ai():
    """
    Test that Agentic AI becomes available
    after both Python and Machine Learning are completed.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Topic 3 requires Topic 1 and Topic 2.
        # Both prerequisites are now completed.
        result = are_prerequisites_completed(
            db=db,
            learning_path_id=1,
            topic_id=3
        )

        # All prerequisites should now be completed
        assert result is True

    finally:
        # Always close the database session after the test
        db.close()