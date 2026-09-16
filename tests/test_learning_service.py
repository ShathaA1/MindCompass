import uuid

from app.database.connection import SessionLocal
from app.services.learning_service import (
    get_active_learning_path,
    select_next_topic,
    are_prerequisites_completed,
    get_topic_mastery,
    get_required_topic_ids,
    create_learning_path,
    get_initial_topic_status,
)

from app.database.models import (
    User,
    LearningPath,
    LearningPathItem,
    TopicMastery,
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


def test_get_topic_mastery():
    """
    Test that the learner's mastery information
    is loaded correctly for the current topic.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Get mastery information for user 2
        # on topic 3: Introduction to Agentic AI.
        result = get_topic_mastery(
            db=db,
            user_id=2,
            topic_id=3
        )

        # Make sure a mastery record was found
        assert result != {}

        # Verify that the returned mastery record
        # belongs to the correct learner and topic
        assert result["user_id"] == 2
        assert result["topic_id"] == 3

        # Verify the test mastery score
        assert result["mastery_score"] == 30

        # Verify the stored weak areas
        assert result["weak_areas"] == [
            "agent planning",
            "tool use"
        ]

    finally:
        # Always close the database session
        db.close()


def test_get_topic_mastery_not_found():
    """
    Test that an empty dictionary is returned
    when no mastery record exists.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Use a user/topic combination that does not
        # have a TopicMastery record in the test data.
        result = get_topic_mastery(
            db=db,
            user_id=2,
            topic_id=999999
        )

        # No mastery record should be returned
        assert result == {}

    finally:
        # Always close the database session
        db.close()


def test_get_required_topic_ids_for_agentic_ai():
    """
    Test that all prerequisites for Agentic AI
    are returned in the correct learning order.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # Get the complete prerequisite chain
        # for Introduction to Agentic AI.
        topic_ids = get_required_topic_ids(
            db=db,
            topic_id=3
        )

        # Python must come first, followed by
        # Machine Learning and then Agentic AI.
        assert topic_ids == [1, 2, 3]


def test_get_required_topic_ids_without_prerequisites():
    """
    Test a topic that does not require
    any prerequisite topics.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # Python Basics has no prerequisites,
        # so only the topic itself should be returned.
        topic_ids = get_required_topic_ids(
            db=db,
            topic_id=1
        )

        assert topic_ids == [1]


def test_create_learning_path():
    """
    Test creating a complete learning path
    from a target topic and its prerequisites.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # Generate a unique email for every test run
        # to avoid conflicts with previous test data.
        unique_email = (
            f"learningpath.test.{uuid.uuid4().hex}"
            "@mindcompass.local"
        )

        # Create a temporary learner
        # so existing test data is not modified.
        test_user = User(
            name="Learning Path Test User",
            email=unique_email,
            password_hash="test_hash",
            role="learner"
        )

        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        # Store the user ID separately so cleanup
        # can still use it if the session fails later.
        test_user_id = test_user.user_id

        try:
            # Create an Agentic AI learning path.
            result = create_learning_path(
                db=db,
                user_id=test_user_id,
                target_topic_id=3,
                path_name="Agentic AI Test Path",
                goal="Learn Agentic AI"
            )

            # Verify the main learning path information.
            assert result["user_id"] == test_user_id
            assert result["name"] == "Agentic AI Test Path"
            assert result["goal"] == "Learn Agentic AI"
            assert result["status"] == "active"

            # Verify that the prerequisite chain
            # was included in the correct order.
            assert result["topic_ids"] == [1, 2, 3]

            # Load the created learning path items
            # directly from the database.
            items = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id
                    == result["learning_path_id"]
                )
                .order_by(
                    LearningPathItem.position.asc()
                )
                .all()
            )

            # Verify that all required topics
            # were added to the learning path.
            assert len(items) == 3

            # Verify the topic order.
            assert [
                item.topic_id
                for item in items
            ] == [
                1,
                2,
                3,
            ]

            # Verify the learning path positions.
            assert [
                item.position
                for item in items
            ] == [
                1,
                2,
                3,
            ]

            # New learning path items should
            # initially be pending.
            assert all(
                item.status == "pending"
                for item in items
            )

            # New topics should initially recommend
            # an explanation until mastery is evaluated.
            assert all(
                item.recommended_action == "explain"
                for item in items
            )

        finally:
            # Reset the session in case a database
            # operation failed during the test.
            db.rollback()

            # Find all learning paths created
            # for the temporary learner.
            learning_paths = (
                db.query(LearningPath)
                .filter(
                    LearningPath.user_id == test_user_id
                )
                .all()
            )

            # Delete the learning path items first
            # to respect foreign key relationships.
            for learning_path in learning_paths:
                (
                    db.query(LearningPathItem)
                    .filter(
                        LearningPathItem.learning_path_id
                        == learning_path.learning_path_id
                    )
                    .delete(
                        synchronize_session=False
                    )
                )

                # Delete the temporary learning path.
                db.delete(learning_path)

            # Find the temporary learner again
            # before deleting it.
            temporary_user = (
                db.query(User)
                .filter(
                    User.user_id == test_user_id
                )
                .first()
            )

            # Delete the temporary learner
            # if it still exists.
            if temporary_user:
                db.delete(temporary_user)

            # Save the cleanup changes.
            db.commit()


def test_get_initial_topic_status_for_low_mastery():
    """
    Test that a topic with low mastery
    remains pending.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # User 2 currently has a mastery score
        # of 30 for Agentic AI.
        status = get_initial_topic_status(
            db=db,
            user_id=2,
            topic_id=3
        )

        assert status == "pending"


def test_get_initial_topic_status_without_mastery():
    """
    Test that a topic without mastery data
    starts as pending.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        status = get_initial_topic_status(
            db=db,
            user_id=2,
            topic_id=1
        )

        assert status == "pending"


def test_get_initial_topic_status_for_high_mastery():
    """
    Test that a topic with high mastery
    starts as completed.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # Create a temporary learner
        # for this test only.
        unique_email = (
            f"mastery.test.{uuid.uuid4().hex}"
            "@mindcompass.local"
        )

        test_user = User(
            name="Mastery Test User",
            email=unique_email,
            password_hash="test_hash",
            role="learner"
        )

        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        # Store the user ID separately
        # for safe cleanup.
        test_user_id = test_user.user_id

        try:
            # Create a high mastery record
            # for Python Basics.
            mastery = TopicMastery(
                user_id=test_user_id,
                topic_id=1,
                mastery_score=90,
                weak_areas=[]
            )

            db.add(mastery)
            db.commit()

            # Determine the initial topic status
            # based on the mastery score.
            status = get_initial_topic_status(
                db=db,
                user_id=test_user_id,
                topic_id=1
            )

            # A strongly mastered topic should
            # already be considered completed.
            assert status == "completed"

        finally:
            # Reset the session if an operation
            # failed during the test.
            db.rollback()

            # Delete the temporary mastery record.
            (
                db.query(TopicMastery)
                .filter(
                    TopicMastery.user_id
                    == test_user_id
                )
                .delete(
                    synchronize_session=False
                )
            )

            # Delete the temporary learner.
            temporary_user = (
                db.query(User)
                .filter(
                    User.user_id == test_user_id
                )
                .first()
            )

            if temporary_user:
                db.delete(temporary_user)

            db.commit()


def test_create_personalized_learning_path():
    """
    Test creating a personalized learning path
    based on the learner's existing topic mastery.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # Generate a unique email for every test run
        # to avoid conflicts with existing test data.
        unique_email = (
            f"personalized.path.{uuid.uuid4().hex}"
            "@mindcompass.local"
        )

        # Create a temporary learner
        # for this test only.
        test_user = User(
            name="Personalized Path Test User",
            email=unique_email,
            password_hash="test_hash",
            role="learner"
        )

        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        # Store the user ID separately
        # for safe cleanup.
        test_user_id = test_user.user_id

        try:
            # Create high mastery records for
            # Python Basics and Machine Learning Basics.
            python_mastery = TopicMastery(
                user_id=test_user_id,
                topic_id=1,
                mastery_score=90,
                weak_areas=[]
            )

            ml_mastery = TopicMastery(
                user_id=test_user_id,
                topic_id=2,
                mastery_score=90,
                weak_areas=[]
            )

            db.add_all([
                python_mastery,
                ml_mastery
            ])
            db.commit()

            # Create an Agentic AI learning path.
            result = create_learning_path(
                db=db,
                user_id=test_user_id,
                target_topic_id=3,
                path_name="Personalized Agentic AI Path",
                goal="Learn Agentic AI"
            )

            # Verify that the full prerequisite chain
            # is still included in the learning path.
            assert result["topic_ids"] == [1, 2, 3]

            # Load the personalized learning path items
            # in their learning order.
            items = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id
                    == result["learning_path_id"]
                )
                .order_by(
                    LearningPathItem.position.asc()
                )
                .all()
            )

            # Verify that all required topics
            # were added to the learning path.
            assert len(items) == 3

            # Verify that Python and ML are already
            # completed because mastery is high.
            assert items[0].topic_id == 1
            assert items[0].status == "completed"

            assert items[1].topic_id == 2
            assert items[1].status == "completed"

            # Verify that Agentic AI remains pending
            # because the learner has no mastery data yet.
            assert items[2].topic_id == 3
            assert items[2].status == "pending"

            # Verify the initial recommended actions.
            assert items[0].recommended_action == "assess"
            assert items[1].recommended_action == "assess"
            assert items[2].recommended_action == "explain"

        finally:
            # Reset the session if a database
            # operation failed during the test.
            db.rollback()

            # Delete temporary learning path items
            # before deleting their learning paths.
            learning_paths = (
                db.query(LearningPath)
                .filter(
                    LearningPath.user_id == test_user_id
                )
                .all()
            )

            for learning_path in learning_paths:
                (
                    db.query(LearningPathItem)
                    .filter(
                        LearningPathItem.learning_path_id
                        == learning_path.learning_path_id
                    )
                    .delete(
                        synchronize_session=False
                    )
                )

                db.delete(learning_path)

            # Delete temporary mastery records.
            (
                db.query(TopicMastery)
                .filter(
                    TopicMastery.user_id == test_user_id
                )
                .delete(
                    synchronize_session=False
                )
            )

            # Delete the temporary learner.
            temporary_user = (
                db.query(User)
                .filter(
                    User.user_id == test_user_id
                )
                .first()
            )

            if temporary_user:
                db.delete(temporary_user)

            # Save all cleanup operations.
            db.commit()