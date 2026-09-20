from sqlalchemy.orm import Session

from app.agent.state import TutorState
from app.database.connection import SessionLocal
from app.services.learner_service import get_learner_context
from app.services.learning_service import (
    get_active_learning_path,
    get_topic_mastery,
    select_next_topic,
    update_learning_path,
    complete_learning_path,
)
from langgraph.graph import StateGraph, START, END

from app.agent.planning import (
    determine_learner_need,
    plan_next_topic,
    recommend_next_action,
    resolve_action,
    route_action,
    route_recommended_action,
)

from app.services.chat_service import get_recent_messages

def load_learner_context(
    state: TutorState,
    db: Session
) -> dict:
    """
    Load the learner's profile information
    from the database and add it to TutorState.
    """

    # Get the learner information using
    # the user_id stored in TutorState.
    learner_context = get_learner_context(
        db=db,
        user_id=state["user_id"]
    )

    # Return the new state field.
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
    # using the learner's user_id.
    learning_path = get_active_learning_path(
        db=db,
        user_id=state["user_id"]
    )

    # Return the new state field.
    return {
        "learning_path": learning_path
    }


def load_topic_mastery(
    state: TutorState,
    db: Session
) -> dict:
    """
    Load the learner's mastery information
    for the current topic and add it to TutorState.
    """

    # Get the currently selected topic
    # from the agent state.
    current_topic = state.get(
        "current_topic",
        {}
    )

    # Get the topic ID needed to load mastery data.
    topic_id = current_topic.get(
        "topic_id"
    )

    # If no current topic has been selected yet,
    # there is no mastery information to load.
    if not topic_id:
        return {
            "topic_mastery": {}
        }

    # Load mastery information for
    # the selected topic.
    topic_mastery = get_topic_mastery(
        db=db,
        user_id=state["user_id"],
        topic_id=topic_id
    )

    # Return the mastery information.
    return {
        "topic_mastery": topic_mastery
    }


def load_conversation_history(
    state: TutorState,
    db: Session
) -> dict:
    """
    Load recent messages from the current chat session
    and add them to TutorState as short-term memory.
    """

    # Get the current chat session ID
    # from the shared agent state.
    session_id = state.get("session_id")

    # If no session is available, return
    # an empty conversation history.
    if not session_id:
        return {
            "conversation_history": []
        }

    # Load the most recent messages
    # from the current chat session.
    conversation_history = get_recent_messages(
        db=db,
        session_id=session_id,
        limit=10
    )

    # Store the recent conversation context
    # in the shared agent state.
    return {
        "conversation_history": conversation_history
    }

# ------------------------------------------------------------------
# LangGraph-compatible node wrappers
# ------------------------------------------------------------------


def load_learner_context_node(
    state: TutorState
) -> dict:
    """
    LangGraph node that loads learner context
    using its own database session.
    """

    # Open a database session for this node execution.
    with SessionLocal() as db:
        return load_learner_context(
            state=state,
            db=db
        )


def load_active_learning_path_node(
    state: TutorState
) -> dict:
    """
    LangGraph node that loads the learner's
    active learning path.
    """

    # Open a database session for this node execution.
    with SessionLocal() as db:
        return load_active_learning_path(
            state=state,
            db=db
        )


def plan_next_topic_node(
    state: TutorState
) -> dict:
    """
    LangGraph node that selects the learner's
    next available topic.
    """

    # Open a database session because
    # topic planning requires database access.
    with SessionLocal() as db:
        return plan_next_topic(
            state=state,
            db=db
        )


def load_topic_mastery_node(
    state: TutorState
) -> dict:
    """
    LangGraph node that loads mastery data
    for the currently selected topic.
    """

    # Open a database session for this node execution.
    with SessionLocal() as db:
        return load_topic_mastery(
            state=state,
            db=db
        )


def load_conversation_history_node(
    state: TutorState
) -> dict:
    """
    LangGraph node that loads recent conversation
    history for the current chat session.
    """

    # Open a database session for
    # loading conversation memory.
    with SessionLocal() as db:
        return load_conversation_history(
            state=state,
            db=db
        )


def build_tutor_graph():
    """
    Build and compile the Tutor Agent workflow.
    """

    # Create a new LangGraph workflow
    # using TutorState as the shared agent state.
    workflow = StateGraph(TutorState)

    # Add the nodes responsible for loading
    # learner and learning path information.
    workflow.add_node(
        "load_learner_context",
        load_learner_context_node
    )

    workflow.add_node(
        "load_conversation_history",
        load_conversation_history_node
    )

    workflow.add_node(
        "load_active_learning_path",
        load_active_learning_path_node
    )

    # Add the node that determines
    # what the learner currently needs.
    workflow.add_node(
        "determine_learner_need",
        determine_learner_need
    )

    # Add the node that selects the next topic
    # based on path order and prerequisites.
    workflow.add_node(
        "plan_next_topic",
        plan_next_topic_node
    )

    # Add the node that loads mastery information
    # for the selected topic.
    workflow.add_node(
        "load_topic_mastery",
        load_topic_mastery_node
    )

    # Add the node that recommends the next action
    # based on the learner's mastery score.
    workflow.add_node(
        "recommend_next_action",
        recommend_next_action
    )

    # Add the node that resolves the final action
    # using both learner intent and recommendation.
    workflow.add_node(
        "resolve_action",
        resolve_action
    )

    # Add placeholder action nodes.
    # These will later be replaced with the real
    # teaching, assessment, practice, and review logic.
    workflow.add_node(
        "teach",
        teach_node
    )

    workflow.add_node(
        "assess",
        assess_node
    )

    workflow.add_node(
        "practice",
        practice_node
    )

    workflow.add_node(
        "review",
        review_node
    )

    # Add the recommendation action node
    # to the Tutor Agent workflow.
    workflow.add_node(
        "recommend",
        recommend_node
    )

    # Add the learning path update node.
    # This node is used when the learner
    # has completed the current topic.
    workflow.add_node(
        "update_learning_path",
        update_learning_path_node
    )

    # Define the main workflow sequence.
    workflow.add_edge(
        START,
        "load_learner_context"
    )

    workflow.add_edge(
        "load_learner_context",
        "load_conversation_history"
    )

    workflow.add_edge(
        "load_conversation_history",
        "load_active_learning_path"
    )

    workflow.add_edge(
        "load_active_learning_path",
        "determine_learner_need"
    )

    workflow.add_edge(
        "determine_learner_need",
        "plan_next_topic"
    )

    workflow.add_edge(
        "plan_next_topic",
        "load_topic_mastery"
    )

    workflow.add_edge(
        "load_topic_mastery",
        "recommend_next_action"
    )

    workflow.add_edge(
        "recommend_next_action",
        "resolve_action"
    )

    # Route the workflow to the correct action node
    # based on the final recommended action.
    workflow.add_conditional_edges(
        "resolve_action",
        route_recommended_action,
        {
            # Route each final agent decision
            # to its corresponding action node.
            "teach": "teach",
            "assess": "assess",
            "practice": "practice",
            "review": "review",
            "recommend": "update_learning_path",
        }
    )

    # After marking the current topic as completed,
    # continue to the recommendation workflow.
    workflow.add_edge(
        "update_learning_path",
        "recommend"
    )
    # End the workflow after the selected action node runs.
    workflow.add_edge(
        "teach",
        END
    )

    workflow.add_edge(
        "assess",
        END
    )

    workflow.add_edge(
        "practice",
        END
    )

    workflow.add_edge(
        "review",
        END
    )

    workflow.add_edge(
        "recommend",
        END
    )

    # Compile the workflow into an executable graph.
    return workflow.compile()




def teach_node(state: TutorState) -> dict:
    """
    Placeholder node for the teaching workflow.
    """

    # Return a temporary response until the
    # real teaching logic is integrated.
    return {
        "response": "Teaching action selected."
    }


def assess_node(state: TutorState) -> dict:
    """
    Placeholder node for the assessment workflow.
    """

    # Return a temporary response until the
    # real assessment logic is integrated.
    return {
        "response": "Assessment action selected."
    }


def practice_node(state: TutorState) -> dict:
    """
    Placeholder node for the practice workflow.
    """

    # Return a temporary response until the
    # real practice logic is integrated.
    return {
        "response": "Practice action selected."
    }


def review_node(state: TutorState) -> dict:
    """
    Placeholder node for the review workflow.
    """

    # Return a temporary response until the
    # real review logic is integrated.
    return {
        "response": "Review action selected."
    }


def recommend_node(
    state: TutorState
) -> dict:
    """
    Recommend the learner's next topic
    after completing the current topic.
    """

    # Get the active learning path
    # from the shared agent state.
    learning_path = state.get(
        "learning_path",
        {}
    )

    # Read the active learning path ID.
    learning_path_id = learning_path.get(
        "learning_path_id"
    )

    # Stop safely if there is no active
    # learning path available.
    if not learning_path_id:
        return {
            "response": (
                "No active learning path was found."
            )
        }

    # Open a database session to find
    # the learner's next available topic.
    with SessionLocal() as db:

        # Select the next incomplete topic
        # whose prerequisites are completed.
        next_topic = select_next_topic(
            db=db,
            learning_path_id=learning_path_id
        )

        # If no topic remains, mark the entire
        # learning path as completed.
        if not next_topic:
            completed_path = complete_learning_path(
                db=db,
                learning_path_id=learning_path_id
            )

            # Update the learning path information
            # stored in the shared agent state.
            updated_learning_path = {
                **learning_path,
                "status": completed_path.get(
                    "status",
                    "completed"
                ),
            }

            return {
                "learning_path": updated_learning_path,
                "response": (
                    "You have completed all topics "
                    "in your learning path."
                ),
            }

    # Store the newly selected topic in TutorState
    # and provide a recommendation response.
    return {
        "current_topic": next_topic,
        "next_topic_id": next_topic["topic_id"],
        "response": (
            f"Your next recommended topic is "
            f"{next_topic['name']}."
        ),
    }

def update_learning_path_node(
    state: TutorState
) -> dict:
    """
    Update the current learning path item
    using the learner's latest mastery result.
    """

    # Get the learner ID from the shared agent state.
    user_id = state.get("user_id")

    # Get the active learning path
    # loaded earlier in the workflow.
    learning_path = state.get(
        "learning_path",
        {}
    )

    # Get the current topic selected
    # by the planning workflow.
    current_topic = state.get(
        "current_topic",
        {}
    )

    # Stop safely if the required information
    # is not available in the agent state.
    if (
        not user_id
        or not learning_path
        or not current_topic
    ):
        return {}

    # Read the required database identifiers.
    learning_path_id = learning_path.get(
        "learning_path_id"
    )
    topic_id = current_topic.get(
        "topic_id"
    )

    # Stop safely if either identifier is missing.
    if not learning_path_id or not topic_id:
        return {}

    # Open a database session for this graph node.
    with SessionLocal() as db:

        # Update the current learning path item
        # using the learner's latest mastery score.
        updated_item = update_learning_path(
            db=db,
            user_id=user_id,
            learning_path_id=learning_path_id,
            topic_id=topic_id
        )

    # Store the updated item information
    # so later nodes can use it if needed.
    return {
        "current_topic": {
            **current_topic,
            "status": updated_item.get(
                "status",
                current_topic.get("status")
            ),
            "recommended_action": updated_item.get(
                "recommended_action"
            ),
        }
    }