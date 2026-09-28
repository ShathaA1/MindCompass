from sqlalchemy.orm import Session

from app.agent.state import TutorState
from app.services.learning_service import select_next_topic
from app.agent.recommendation import build_recommendation

def determine_learner_need(
    state: TutorState
) -> dict:
    """
    Determine what the learner currently needs
    based on the user message.
    """

    # Normalize the learner message so keyword
    # matching is case-insensitive.
    user_message = state.get(
        "user_message",
        ""
    ).lower()

    # Define learner intents in priority order.
    intent_keywords = {
        "explain": [
            "explain",
            "understand",
            "what is",
            "how does",
            "how do",
            "tell me about",
            "clarify",
        ],
        "assess": [
            "quiz",
            "test me",
            "assess",
            "assessment",
            "questions",
            "check my understanding",
        ],
        "practice": [
            "practice",
            "exercise",
            "example",
            "give me practice",
            "try some",
        ],
        "review": [
            "review",
            "revise",
            "recap",
            "summarize",
            "go over",
        ],
        "recommend": [
            "next",
            "recommend",
            "what should i learn",
            "what should i do",
            "continue",
        ],
    }

    # Check each intent in priority order
    # and stop at the first matching category.
    for learner_need, keywords in intent_keywords.items():
        if any(
            keyword in user_message
            for keyword in keywords
        ):
            return {
                "learner_need": learner_need
            }

    # Use recommendation when no explicit
    # learner intent can be identified.
    return {
        "learner_need": "recommend"
    }


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



def recommend_next_action(
    state: TutorState
) -> dict:
    """
    Recommend the learner's next action and explain
    the reason behind the recommendation.
    """

    # Get the learner's mastery information
    # for the current topic.
    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    # Use None when the learner has not completed
    # an assessment for the current topic.
    mastery_score = (
        topic_mastery.get("mastery_score")
        if topic_mastery
        else None
    )

    # Load the learner's identified weak areas.
    # An empty list is used when no weak areas exist.
    weak_areas = (
        topic_mastery.get("weak_areas") or []
        if topic_mastery
        else []
    )


    last_action = state.get(
        "last_action"
    )

    # Build a detailed recommendation using
    # mastery and the learner's weak areas.
    recommendation = build_recommendation(
        mastery_score=mastery_score,
        weak_areas=weak_areas,
        last_action=last_action,
    )

    # Store both the action and its reason
    # in the shared TutorState.
    return recommendation


def resolve_action(
    state: TutorState
) -> dict:
    """
    Resolve the final agent action by combining
    the learner's explicit request with the recommendation.
    """

    # Read the learner's requested action.
    learner_need = state.get(
        "learner_need",
        "recommend"
    )

    # Read the action recommended from mastery data.
    recommended_action = state.get(
        "recommended_action",
        "explain"
    )

    # If the learner explicitly asked for an action,
    # respect the learner's request.
    if learner_need in [
        "explain",
        "assess",
        "practice",
        "review",
    ]:
        final_action = learner_need

    # If the learner asked for a recommendation,
    # use the action selected from mastery data.
    else:
        final_action = recommended_action

    # Store the final decision in TutorState.
    return {
        "recommended_action": final_action
    }


def route_recommended_action(
    state: TutorState
) -> str:
    """
    Route the workflow based on the final
    recommended action stored in TutorState.
    """

    # Read the final action selected after
    # learner intent and mastery are resolved.
    recommended_action = state.get(
        "recommended_action",
        "explain"
    )

    # Map each supported action to its
    # corresponding LangGraph node.
    action_routes = {
        "explain": "teach",
        "assess": "assess",
        "practice": "practice",
        "review": "review",
        "recommend": "recommend",
    }

    # Use teaching as a safe fallback
    # when the action is unknown.
    return action_routes.get(
        recommended_action,
        "teach"
    )