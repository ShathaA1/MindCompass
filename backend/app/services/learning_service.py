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


def get_required_topic_ids(
    db: Session,
    topic_id: int
) -> list[int]:
    """
    Get all prerequisite topics required for a target topic
    in the correct learning order.
    """

    # Store the topics in learning order.
    required_topic_ids = []

    # Keep track of visited topics to avoid
    # processing the same topic more than once.
    visited = set()

    def collect_prerequisites(
        current_topic_id: int
    ) -> None:
        """
        Recursively collect prerequisites
        before adding the current topic.
        """

        # Skip topics that have already been processed.
        if current_topic_id in visited:
            return

        # Mark the topic as visited.
        visited.add(current_topic_id)

        # Get all direct prerequisites
        # required for the current topic.
        prerequisites = (
            db.query(TopicPrerequisite)
            .filter(
                TopicPrerequisite.topic_id
                == current_topic_id
            )
            .all()
        )

        # Process prerequisites first so they appear
        # before the current topic in the learning path.
        for prerequisite in prerequisites:
            collect_prerequisites(
                prerequisite.prerequisite_topic_id
            )

        # Add the current topic only after
        # all of its prerequisites.
        required_topic_ids.append(
            current_topic_id
        )

    # Start from the learner's target topic.
    collect_prerequisites(topic_id)

    return required_topic_ids


def create_learning_path(
    db: Session,
    user_id: int,
    target_topic_id: int,
    path_name: str,
    goal: str
) -> dict:
    """
    Create a new learning path for a learner
    based on the target topic and its prerequisites.
    """

    # Get all topics required to reach the target topic
    # in the correct learning order.
    required_topic_ids = get_required_topic_ids(
        db=db,
        topic_id=target_topic_id
    )

    # Create the learner's new active learning path.
    learning_path = LearningPath(
        user_id=user_id,
        name=path_name,
        goal=goal,
        status="active"
    )

    # Add the learning path to the current transaction
    # so its ID can be generated.
    db.add(learning_path)
    db.flush()

    # Create one learning path item
    # for every required topic.
    for position, topic_id in enumerate(
        required_topic_ids,
        start=1
    ):
        # Determine whether the learner already
        # mastered this topic before creating the path.
        initial_status = get_initial_topic_status(
            db=db,
            user_id=user_id,
            topic_id=topic_id
        )

        # Determine the learner's initial topic status
        # based on the latest assessment result.
        initial_status = get_initial_topic_status(
            db=db,
            user_id=user_id,
            topic_id=topic_id
        )

        # Determine the appropriate learning action
        # using the same recommendation logic used
        # throughout the system.
        recommended_action = get_topic_recommended_action(
            db=db,
            user_id=user_id,
            topic_id=topic_id
        )

        # Create the personalized learning path item.
        learning_path_item = LearningPathItem(
            learning_path_id=learning_path.learning_path_id,
            topic_id=topic_id,
            position=position,
            status=initial_status,
            recommended_action=recommended_action
        )

        db.add(learning_path_item)

    # Save the learning path and all of its items.
    db.commit()

    # Refresh the object with the latest
    # values stored in the database.
    db.refresh(learning_path)

    # Return the created learning path information.
    return {
        "learning_path_id": learning_path.learning_path_id,
        "user_id": learning_path.user_id,
        "name": learning_path.name,
        "goal": learning_path.goal,
        "status": learning_path.status,
        "topic_ids": required_topic_ids,
    }


def get_initial_topic_status(
    db: Session,
    user_id: int,
    topic_id: int
) -> str:
    """
    Determine the initial learning path status
    for a topic based on learner mastery.
    """

    # Get the learner's existing mastery
    # information for this topic.
    topic_mastery = get_topic_mastery(
        db=db,
        user_id=user_id,
        topic_id=topic_id
    )

    # If the learner has never been assessed
    # on this topic, it should remain pending.
    if not topic_mastery:
        return "pending"

    # Read the learner's mastery score.
    mastery_score = topic_mastery.get(
        "mastery_score",
        0
    )

    # Consider strongly mastered topics completed
    # when building the initial learning path.
    if mastery_score >= 85:
        return "completed"

    # Topics below the mastery threshold
    # still need to be studied.
    return "pending"


def update_learning_path(
    db: Session,
    user_id: int,
    learning_path_id: int,
    topic_id: int
) -> dict:
    """
    Update a learning path item based on
    the learner's latest topic mastery.
    """

    # Find the requested topic inside
    # the learner's learning path.
    learning_path_item = (
        db.query(LearningPathItem)
        .join(LearningPath)
        .filter(
            LearningPathItem.learning_path_id
            == learning_path_id,
            LearningPathItem.topic_id
            == topic_id,
            LearningPath.user_id
            == user_id
        )
        .first()
    )

    # Return an empty dictionary if the topic
    # does not exist in this learner's path.
    if not learning_path_item:
        return {}

    # Recalculate the topic status using
    # the learner's latest mastery data.
    updated_status = get_initial_topic_status(
        db=db,
        user_id=user_id,
        topic_id=topic_id
    )

    # Update the learning path item status.
    learning_path_item.status = updated_status

    # Determine the learner's next action
    # using the latest assessment mastery score.
    updated_action = get_topic_recommended_action(
        db=db,
        user_id=user_id,
        topic_id=topic_id
    )

    # Update the learning path item with
    # the latest status and recommended action.
    learning_path_item.status = updated_status
    learning_path_item.recommended_action = updated_action

    # Save the updated learning path item.
    db.commit()
    db.refresh(learning_path_item)

    # Return the updated information so it
    # can later be stored inside TutorState.
    return {
        "learning_path_item_id":
            learning_path_item.learning_path_item_id,
        "learning_path_id":
            learning_path_item.learning_path_id,
        "topic_id":
            learning_path_item.topic_id,
        "status":
            learning_path_item.status,
        "recommended_action":
            learning_path_item.recommended_action,
    }


def complete_learning_path(
    db: Session,
    learning_path_id: int
) -> dict:
    """
    Mark a learning path as completed
    after all of its topics are completed.
    """

    # Find the requested learning path
    # in the database.
    learning_path = (
        db.query(LearningPath)
        .filter(
            LearningPath.learning_path_id
            == learning_path_id
        )
        .first()
    )

    # Return an empty dictionary if
    # the learning path does not exist.
    if not learning_path:
        return {}

    # Mark the learning path as completed.
    learning_path.status = "completed"

    # Save the updated learning path.
    db.commit()

    # Refresh the object with the latest
    # values stored in the database.
    db.refresh(learning_path)

    # Return the updated learning path information.
    return {
        "learning_path_id": (
            learning_path.learning_path_id
        ),
        "user_id": learning_path.user_id,
        "name": learning_path.name,
        "goal": learning_path.goal,
        "status": learning_path.status,
    }


def get_topic_recommended_action(
    db: Session,
    user_id: int,
    topic_id: int
) -> str:
    """
    Determine the recommended learning action
    based on the learner's latest assessment score.
    """

    # Get the learner's latest mastery information
    # for the requested topic.
    topic_mastery = get_topic_mastery(
        db=db,
        user_id=user_id,
        topic_id=topic_id
    )

    # If the learner has not completed an assessment
    # yet, start by explaining the topic.
    if not topic_mastery:
        return "explain"

    # Mastery score represents the result
    # of the learner's latest assessment.
    mastery_score = topic_mastery.get(
        "mastery_score",
        0
    )

    # Very low mastery requires explanation.
    if mastery_score < 40:
        return "explain"

    # Moderate mastery requires more practice.
    if mastery_score < 70:
        return "practice"

    # Good mastery requires targeted review
    # of the learner's remaining weak areas.
    if mastery_score < 85:
        return "review"

    # High mastery means the topic is completed,
    # so the agent should determine the next step.
    return "recommend"