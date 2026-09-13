from sqlalchemy.orm import Session

from app.agent.state import TutorState
from app.services.learner_service import get_learner_context
from app.services.learning_service import get_active_learning_path


def load_learner_context(
    state: TutorState,
    db: Session
) -> dict:
    """
    Load the learner's profile information
    from the database and add it to TutorState.
    """

    # Get the learner information using
    # the user_id stored in TutorState
    learner_context = get_learner_context(
        db=db,
        user_id=state["user_id"]
    )

    # Return only the new state field
    # so it can be merged into TutorState
    return {
        "learner_context": learner_context
    }


def load_active_learning_path(
    state: TutorState,
    db: Session
) -> dict:
    """
    Load the learner's active learning path
    and add it to TutorState.
    """

    # Get the active learning path
    # using the learner's user_id
    learning_path = get_active_learning_path(
        db=db,
        user_id=state["user_id"]
    )

    # Return only the new state field
    # so it can be merged into TutorState
    return {
        "learning_path": learning_path
    }