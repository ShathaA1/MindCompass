from typing import TypedDict, Any


class TutorState(TypedDict, total=False):
    # Current user and session
    user_id: int
    session_id: int
    user_message: str

    # Short-term conversation memory
    conversation_history: list[dict[str, Any]]

    # Learner data loaded from the database
    learner_context: dict[str, Any]
    learning_path: dict[str, Any]
    current_topic: dict[str, Any]
    topic_mastery: dict[str, Any]

    # Agent decision
    learner_need: str
    recommended_action: str
    recommendation_reason: str
    next_topic_id: int

    # Final result from other nodes/tools
    response: str