from sqlalchemy.orm import Session

from app.agent.state import TutorState
from app.services.learning_service import select_next_topic


def determine_learner_need(
    state: TutorState
) -> dict:
    """
    Determine what the learner currently needs
    based on the user message.
    """

    # Convert the user message to lowercase
    # to make keyword matching case-insensitive
    user_message = state.get(
        "user_message",
        ""
    ).lower()

    # If the learner asks for an explanation,
    # choose the explain action
    if any(
        keyword in user_message
        for keyword in [
            "explain",
            "understand",
            "what is",
            "how does",
        ]
    ):
        learner_need = "explain"

    # If the learner asks for a quiz or assessment,
    # choose the assess action
    elif any(
        keyword in user_message
        for keyword in [
            "quiz",
            "test me",
            "assess",
            "assessment",
        ]
    ):
        learner_need = "assess"

    # If the learner asks to practice,
    # choose the practice action
    elif any(
        keyword in user_message
        for keyword in [
            "practice",
            "exercise",
            "example",
        ]
    ):
        learner_need = "practice"

    # If the learner asks to review previous material,
    # choose the review action
    elif any(
        keyword in user_message
        for keyword in [
            "review",
            "revise",
            "recap",
        ]
    ):
        learner_need = "review"

    # If the learner asks what to do next,
    # choose the recommendation action
    elif any(
        keyword in user_message
        for keyword in [
            "next",
            "recommend",
            "what should i learn",
        ]
    ):
        learner_need = "recommend"

    # Use recommendation as the default action
    # when the request does not match another category
    else:
        learner_need = "recommend"

    # Return the decision so LangGraph can merge it
    # into the existing TutorState
    return {
        "learner_need": learner_need
    }


def route_action(
    state: TutorState
) -> str:
    """
    Route the workflow based on the learner need
    already determined by the agent.
    """

    # Read the learner need from TutorState
    learner_need = state.get(
        "learner_need",
        "recommend"
    )

    # Route explanation requests
    # to the teaching node
    if learner_need == "explain":
        return "teach"

    # Route assessment requests
    # to the assessment node
    if learner_need == "assess":
        return "assess"

    # Route practice requests
    # to the practice node
    if learner_need == "practice":
        return "practice"

    # Route review requests
    # to the review node
    if learner_need == "review":
        return "review"

    # Route recommendation requests and unknown values
    # to the recommendation node
    return "recommend"


def plan_next_topic(
    state: TutorState,
    db: Session
) -> dict:
    """
    Select the next available topic for the learner
    based on the active learning path and prerequisites.
    """

    # Get the active learning path
    # already loaded into TutorState
    learning_path = state.get(
        "learning_path",
        {}
    )

    # Get the learning path ID
    # needed to select the next topic
    learning_path_id = learning_path.get(
        "learning_path_id"
    )

    # If the learner does not have an active learning path,
    # there is no topic to plan
    if not learning_path_id:
        return {
            "current_topic": {}
        }

    # Select the next incomplete topic whose
    # prerequisites have already been completed
    next_topic = select_next_topic(
        db=db,
        learning_path_id=learning_path_id
    )

    # Store the selected topic in TutorState
    # so it can be used by later agent nodes
    return {
        "current_topic": next_topic
    }