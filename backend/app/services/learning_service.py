from sqlalchemy.orm import Session

from app.database.models import (
    LearningPath,
    LearningPathItem,
    Topic,
    TopicPrerequisite,
    TopicMastery,
)

from app.agent.recommendation import determine_recommended_action

from app.core.learning_paths import AVAILABLE_LEARNING_PATHS


def get_available_learning_path(path_key: str) -> dict:
    """
    Validate the learner's selected learning path
    and return its configuration.
    """

    # Normalize the path key before validation.
    normalized_path_key = path_key.strip().lower()

    # Make sure the selected path exists
    # in the platform configuration.
    if normalized_path_key not in AVAILABLE_LEARNING_PATHS:
        raise ValueError(
            f"Unsupported learning path: {path_key}"
        )

    # Return the selected learning path configuration.
    return AVAILABLE_LEARNING_PATHS[normalized_path_key]

def get_diagnostic_topics(
    db: Session,
    path_key: str
) -> list[dict]:
    """
    Load the topics that should be assessed
    before the learner starts the selected learning path.
    """

    # Validate the selected learning path
    # and load its configuration.
    path_config = get_available_learning_path(path_key)

    # Get the prerequisite topics that should
    # be included in the diagnostic assessment.
    diagnostic_topic_ids = path_config.get(
        "diagnostic_topic_ids",
        []
    )

    # Some learning paths may not require
    # a prerequisite diagnostic assessment.
    if not diagnostic_topic_ids:
        return []

    # Load the diagnostic topics from the database.
    topics = (
        db.query(Topic)
        .filter(Topic.topic_id.in_(diagnostic_topic_ids))
        .all()
    )

    # Create a lookup so the final result follows
    # the order defined in the path configuration.
    topics_by_id = {
        topic.topic_id: topic
        for topic in topics
    }

    # Make sure every configured topic exists
    # in the database.
    missing_topic_ids = [
        topic_id
        for topic_id in diagnostic_topic_ids
        if topic_id not in topics_by_id
    ]

    if missing_topic_ids:
        raise ValueError(
            f"Diagnostic topics not found: {missing_topic_ids}"
        )

    # Return the structure expected by
    # the quiz generation tool.
    return [
        {
            "topic_id": topic_id,
            "topic": topics_by_id[topic_id].name,
        }
        for topic_id in diagnostic_topic_ids
    ]


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
    are satisfied in the learner's personalized learning path.
    """

    # Topics 1-3 are structural/diagnostic topics.
    # They are not included as study items in personalized paths.
    structural_topic_ids = {1, 2, 3}

    # Get all prerequisite topic IDs required for this topic.
    prerequisites = (
        db.query(TopicPrerequisite)
        .filter(
            TopicPrerequisite.topic_id == topic_id
        )
        .all()
    )

    # A topic with no prerequisites can be studied immediately.
    if not prerequisites:
        return True

    # Check every prerequisite one by one.
    for prerequisite in prerequisites:
        prerequisite_topic_id = prerequisite.prerequisite_topic_id

        # Structural prerequisites are handled by the initial
        # diagnostic and personalized path construction.
        if prerequisite_topic_id in structural_topic_ids:
            continue

        # For real learning-content prerequisites,
        # the prerequisite must exist in the same learning path.
        prerequisite_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == learning_path_id,
                LearningPathItem.topic_id == prerequisite_topic_id
            )
            .first()
        )

        # A real learning prerequisite must exist
        # and must already be completed.
        if (
            not prerequisite_item
            or prerequisite_item.status != "completed"
        ):
            return False

    # All prerequisites are satisfied.
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


def get_personalized_topic_ids(
    db: Session,
    user_id: int,
    target_topic_id: int
) -> list[int]:
    """
    Build the learner's personalized study-topic list
    based on diagnostic mastery and available learning content.
    """

    # Topic IDs used only to store diagnostic mastery.
    python_diagnostic_topic_id = 1
    ml_diagnostic_topic_id = 2

    # RAG-backed reference materials used to fill
    # prerequisite knowledge gaps.
    python_material_topic_id = 27
    ml_material_topic_id = 26

    # Topics 1-3 are diagnostic/structural records
    # and should not become study items.
    non_learning_topic_ids = {1, 2, 3}

    personalized_topic_ids = []

    # Check Python readiness from the initial diagnostic.
    python_mastery = get_topic_mastery(
        db=db,
        user_id=user_id,
        topic_id=python_diagnostic_topic_id
    )

    python_score = (
        python_mastery.get("mastery_score")
        if python_mastery
        else None
    )

    # Add the actual Python learning material
    # only when the learner has a Python gap.
    if python_score is None or python_score < 85:
        personalized_topic_ids.append(
            python_material_topic_id
        )

    # Check Machine Learning readiness
    # from the initial diagnostic.
    ml_mastery = get_topic_mastery(
        db=db,
        user_id=user_id,
        topic_id=ml_diagnostic_topic_id
    )

    ml_score = (
        ml_mastery.get("mastery_score")
        if ml_mastery
        else None
    )

    # Add the actual ML learning material
    # only when the learner has an ML gap.
    if ml_score is None or ml_score < 85:
        personalized_topic_ids.append(
            ml_material_topic_id
        )

    # Load the curriculum required to reach
    # the selected target topic.
    required_topic_ids = get_required_topic_ids(
        db=db,
        topic_id=target_topic_id
    )

    # Add only real learning-content topics.
    # Diagnostic/structural topics are excluded.
    personalized_topic_ids.extend(
        topic_id
        for topic_id in required_topic_ids
        if topic_id not in non_learning_topic_ids
    )

    return personalized_topic_ids


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

    # Build a personalized list of study topics based on
    # the learner's diagnostic mastery and curriculum target.
    required_topic_ids = get_personalized_topic_ids(
        db=db,
        user_id=user_id,
        target_topic_id=target_topic_id
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
    using the shared recommendation rules.
    """

    # Get the learner's latest mastery information
    # for the requested topic.
    topic_mastery = get_topic_mastery(
        db=db,
        user_id=user_id,
        topic_id=topic_id
    )

    # Use None when the learner has not completed
    # an assessment for this topic yet.
    mastery_score = (
        topic_mastery.get("mastery_score")
        if topic_mastery
        else None
    )

    # Apply the shared recommendation rules
    # used across the Tutor Agent.
    return determine_recommended_action(
        mastery_score=mastery_score
    )

