from sqlalchemy.orm import Session

from app.database.models import (
    LearningPath,
    LearningPathItem,
    Topic,
    TopicPrerequisite,
    TopicMastery,
)


def get_active_learning_path(
    db: Session,
    user_id: int
) -> dict:
    """
    Get the learner's currently active learning path.
    """

    # Search for an active learning path
    # that belongs to this learner
    learning_path = (
        db.query(LearningPath)
        .filter(
            LearningPath.user_id == user_id,
            LearningPath.status == "active"
        )
        .first()
    )

    # Return an empty dictionary if the learner
    # does not have an active learning path
    if not learning_path:
        return {}

    # Convert the database object into a simple dictionary
    # that can be stored inside TutorState
    return {
        "learning_path_id": learning_path.learning_path_id,
        "user_id": learning_path.user_id,
        "name": learning_path.name,
        "goal": learning_path.goal,
        "status": learning_path.status,
    }


def are_prerequisites_completed(
    db: Session,
    learning_path_id: int,
    topic_id: int
) -> bool:
    """
    Check whether all prerequisites for a topic
    are completed in the learner's current learning path.
    """

    # Get all prerequisite topic IDs required for this topic
    prerequisites = (
        db.query(TopicPrerequisite)
        .filter(
            TopicPrerequisite.topic_id == topic_id
        )
        .all()
    )

    # If the topic has no prerequisites,
    # the learner can study it immediately
    if not prerequisites:
        return True

    # Check every prerequisite one by one
    for prerequisite in prerequisites:

        # Find the prerequisite topic
        # inside the same learning path
        prerequisite_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == learning_path_id,
                LearningPathItem.topic_id
                == prerequisite.prerequisite_topic_id
            )
            .first()
        )

        # If the prerequisite is missing from the learning path,
        # or it has not been completed,
        # the topic is not ready yet
        if (
            not prerequisite_item
            or prerequisite_item.status != "completed"
        ):
            return False

    # All prerequisites were found and completed
    return True


def select_next_topic(
    db: Session,
    learning_path_id: int
) -> dict:
    """
    Select the next incomplete topic in the learning path
    whose prerequisites have already been completed.
    """

    # Load all incomplete items in the learning path
    # and keep them ordered by their defined position
    incomplete_items = (
        db.query(LearningPathItem)
        .filter(
            LearningPathItem.learning_path_id == learning_path_id,
            LearningPathItem.status != "completed"
        )
        .order_by(LearningPathItem.position.asc())
        .all()
    )

    # Check each incomplete topic in sequence
    for item in incomplete_items:

        # Skip this topic if its prerequisites
        # have not been completed yet
        if not are_prerequisites_completed(
            db=db,
            learning_path_id=learning_path_id,
            topic_id=item.topic_id
        ):
            continue

        # Load the Topic details
        # for the first valid learning path item
        topic = (
            db.query(Topic)
            .filter(
                Topic.topic_id == item.topic_id
            )
            .first()
        )

        # Skip invalid learning path items
        # whose Topic record cannot be found
        if not topic:
            continue

        # Return the first incomplete topic
        # that is ready for the learner
        return {
            "learning_path_item_id": item.learning_path_item_id,
            "topic_id": topic.topic_id,
            "name": topic.name,
            "description": topic.description,
            "difficulty_level": topic.difficulty_level,
            "position": item.position,
            "status": item.status,
            "recommended_action": item.recommended_action,
        }

    # Return an empty dictionary if there is
    # no available topic to study next
    return {}



def get_topic_mastery(
    db: Session,
    user_id: int,
    topic_id: int
) -> dict:
    """
    Get the learner's mastery information
    for a specific topic.
    """

    # Search for the learner's mastery record
    # for the requested topic.
    mastery = (
        db.query(TopicMastery)
        .filter(
            TopicMastery.user_id == user_id,
            TopicMastery.topic_id == topic_id
        )
        .first()
    )

    # Return an empty dictionary if the learner
    # has not been assessed on this topic yet.
    if not mastery:
        return {}

    # Convert the database record into a dictionary
    # that can be stored inside TutorState.
    return {
        "topic_mastery_id": mastery.topic_mastery_id,
        "user_id": mastery.user_id,
        "topic_id": mastery.topic_id,
        "mastery_score": mastery.mastery_score,
        "weak_areas": mastery.weak_areas,
        "last_assessed_at": mastery.last_assessed_at,
    }