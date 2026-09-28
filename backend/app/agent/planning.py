from sqlalchemy.orm import Session

from app.agent.state import TutorState
from app.services.learning_service import select_next_topic
from app.agent.recommendation import build_recommendation

from app.agent.intent_classifier import (
    classify_learner_intent,
)

def determine_learner_need(
    state: TutorState
) -> dict:
    """
    Determine what the learner currently needs
    based on the learner's message.
    """

    # Normalize the learner message so intent
    # matching is case-insensitive.
    user_message = state.get(
        "user_message",
        ""
    ).lower().strip()

    # Define learner intents in priority order.
    # More specific learning requests should be
    # checked before general continuation requests.
    # Define learner intents in priority order.
    intent_keywords = {
        "assess": [
            "quiz me",
            "test me",
            "start assessment",
            "take assessment",
        ],
        "practice": [
            "start practice",
            "give me practice",
            "let's practice",
        ],
        "review": [
            "start review",
            "review this topic",
            "review my weak areas",
        ],
        "recommend": [
            "what should i do next",
            "what's next",
            "continue",
        ],
    }

    # Check each intent in priority order and
    # stop at the first matching category.
    for learner_need, keywords in intent_keywords.items():
        if any(
            keyword in user_message
            for keyword in keywords
        ):
            return {
                "learner_need": learner_need
            }

    # Use the LLM classifier when keyword matching
    # cannot confidently identify the learner intent.
    conversation_history = state.get(
        "conversation_history",
        []
    )

    learner_need = classify_learner_intent(
        user_message=user_message,
        conversation_history=conversation_history,
    )

    return {
        "learner_need": learner_need
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
    Resolve the final agent action while separating
    recommendation requests from action execution.
    """

    # Read the learner's requested action.
    learner_need = state.get("learner_need")

    # Preserve the action selected from mastery data.
    suggested_action = state.get(
        "recommended_action",
        "explain"
    )

    # Explicit execution requests should run immediately.
    if learner_need in [
        "explain",
        "assess",
        "practice",
        "review",
    ]:
        final_action = learner_need

    # A recommendation request should display the
    # suggested action and reason without executing it.
    elif learner_need == "recommend":
        final_action = "recommend"

    # Preserve the mastery-based recommendation when
    # no explicit learner intent is available.
    else:
        final_action = suggested_action

    return {
        "recommended_action": final_action,
        "suggested_action": suggested_action,
    }


def route_recommended_action(
    state: TutorState
) -> str:
    """
    Route recommendation questions separately from
    requests that execute a learning action.
    """

    learner_need = state.get("learner_need")

    # Display the recommendation without executing it.
    if learner_need == "recommend":
        return "recommend_action"

    recommended_action = state.get(
        "recommended_action",
        "explain"
    )

    action_routes = {
        "explain": "teach",
        "assess": "assess",
        "practice": "practice",
        "review": "review",
        "recommend": "recommend",
    }

    return action_routes.get(
        recommended_action,
        "teach"
    )