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
        "explain": [
            "explain",
            "explain this",
            "explain it",
            "explain again",
            "explain more",
            "can you explain",
            "could you explain",
            "please explain",
            "understand",
            "help me understand",
            "i don't understand",
            "i do not understand",
            "i'm confused",
            "i am confused",
            "clarify",
            "simplify",
            "make it simpler",
            "in simple words",
            "in simple terms",
            "what is",
            "what are",
            "what does",
            "what do",
            "what does this mean",
            "define",
            "definition",
            "tell me about",
            "teach me",
            "how does",
            "how do",
            "how is",
            "how can",
            "why",
            "why is",
            "why does",
            "difference between",
            "what is the difference",
            "compare",
            "example",
            "examples",
            "give me an example",
            "give me examples",
            "give me more examples",
            "more examples",
            "another example",
            "show me an example",
            "show me examples",
            "application",
            "applications",
            "more applications",
            "another application",
            "use case",
            "use cases",
            "real-world example",
            "real world example",
            "tell me more",
            "more details",
            "more information",
            "elaborate",
            "expand on",
            "what about",
            "how about",
            "more about",
            "another example",
            "another application",
            "another field",
            "another area",
            "another domain",
            "other examples",
            "other applications",
            "other fields",
            "other areas",
            "other domains",
        ],

        "assess": [
            "quiz me",
            "give me a quiz",
            "start quiz",
            "take a quiz",
            "test me",
            "give me a test",
            "assess me",
            "start assessment",
            "take assessment",
            "take an assessment",
            "check my understanding",
            "check my knowledge",
            "evaluate me",
            "test my understanding",
            "test my knowledge",
            "assessment questions",
            "mini exam",
        ],

        "practice": [
            "practice",
            "practice this",
            "practice the topic",
            "practice this topic",
            "start practice",
            "give me practice",
            "give me some practice",
            "practice activity",
            "practice activities",
            "exercise",
            "exercises",
            "give me an exercise",
            "give me exercises",
            "let me practice",
            "i want to practice",
            "let's practice",
            "try some exercises",
            "hands-on practice",
            "hands on practice",
            "practice questions",
            "practice problem",
            "practice problems",
            "coding practice",
            "coding exercise",
            "scenario practice",
            "flashcards",
            "flashcard",
            "short answer",
            "short-answer",
            "guided practice",
        ],

        "review": [
            "review",
            "review this",
            "review the topic",
            "review this topic",
            "start review",
            "revise",
            "revision",
            "recap",
            "give me a recap",
            "summarize",
            "summarise",
            "summary",
            "give me a summary",
            "go over",
            "go over this",
            "refresh my memory",
            "quick review",
            "quick recap",
            "review weak areas",
            "review my weak areas",
            "review mistakes",
            "review my mistakes",
            "focus on weak areas",
        ],

        "recommend": [
            "what's next",
            "whats next",
            "what is next",
            "next step",
            "what is the next step",
            "what should i do",
            "what should i do next",
            "what should i learn",
            "what should i learn next",
            "what do you recommend",
            "recommend",
            "continue",
            "continue learning",
            "continue the lesson",
            "keep going",
            "go on",
            "move on",
            "move forward",
            "proceed",
            "ready to continue",
            "what now",
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