from datetime import datetime, timezone
import json
import pytest
from app.database.connection import SessionLocal
from app.agent.graph import (
    load_learner_context,
    load_active_learning_path,
    load_topic_mastery,
    build_tutor_graph,
    update_learning_path_node,
    recommend_node,
    recommend_action_node,
    load_conversation_history,
    prepare_assessment_inputs,
    generate_assessment_node,
    prepare_assessment_responses,
    evaluate_assessment_responses,
    submit_assessment_node,
    route_assessment,
    teach_node,
    practice_node,
    review_node,
    generate_initial_diagnostic_node,
    submit_initial_diagnostic_node,
    route_tutor_entry,
    select_practice_type,
    extract_practice_item_count,
    prepare_personalized_inputs,
    route_topic_scope,
    out_of_scope_node,
    plan_next_topic_node,
)
from app.agent.planning import (
    determine_learner_need,
    plan_next_topic,
    recommend_next_action,
    resolve_action,
    route_recommended_action,
)

from app.database.models import (
    TopicMastery,
    LearningPathItem,
    LearningPath,
    ChatSession,
    ChatMessage,
)

from app.services.learning_service import (
    get_active_learning_path,
    get_topic_by_key,
    get_topic_id_by_key,
    select_next_topic,
)

from app.agent.recommendation import build_recommendation

from app.core.learning_paths import (
    AVAILABLE_LEARNING_PATHS,
)

def test_load_learner_context():
    """
    Test that the agent node loads learner data
    from the database and returns it in learner_context.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Initial agent state
        state = {
            "user_id": 2,
            "session_id": 1,
            "user_message": "What should I learn next?"
        }

        # Run the learner context node
        result = load_learner_context(
            state=state,
            db=db
        )

        # Make sure learner_context was returned
        assert "learner_context" in result

        # Verify that the correct learner was loaded
        assert result["learner_context"]["user_id"] == 2
        assert result["learner_context"]["name"] == "Test Learner"

    finally:
        # Always close the database session after the test
        db.close()


def test_load_active_learning_path():
    """
    Test that the agent node loads the learner's
    active learning path from the database.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # The node only needs user_id to find the learning path
        state = {
            "user_id": 2,
            "session_id": 1,
            "user_message": "What should I learn next?"
        }

        # Run the learning path node
        result = load_active_learning_path(
            state=state,
            db=db
        )

        # Make sure learning_path was returned
        assert "learning_path" in result

        # Verify that the correct active path was loaded
        assert result["learning_path"]["learning_path_id"] == 1
        assert result["learning_path"]["user_id"] == 2
        assert result["learning_path"]["name"] == "Agentic AI Learning Path"
        assert result["learning_path"]["status"] == "active"

    finally:
        # Always close the database session after the test
        db.close()

def test_determine_learner_need_explain():
    """
    Test that explanation-related messages
    are classified as explain.
    """

    # Create a minimal TutorState for the test
    state = {
        "user_id": 2,
        "user_message": "Can you explain what an agent is?"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "explain"


def test_determine_learner_need_assess():
    """
    Test that quiz or assessment requests
    are classified as assess.
    """

    # Create a learner message asking for a quiz
    state = {
        "user_id": 2,
        "user_message": "Quiz me on this topic"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "assess"


def test_determine_learner_need_practice():
    """
    Test that practice-related requests
    are classified as practice.
    """

    # Create a learner message asking for practice
    state = {
        "user_id": 2,
        "user_message": "I want to practice this topic"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "practice"


def test_determine_learner_need_review():
    """
    Test that review-related requests
    are classified as review.
    """

    # Create a learner message asking for a review
    state = {
        "user_id": 2,
        "user_message": "Can we review the previous topic?"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "review"


def test_determine_learner_need_recommend():
    """
    Test that next-step requests
    are classified as recommend.
    """

    # Create a learner message asking what to learn next
    state = {
        "user_id": 2,
        "user_message": "What should I learn next?"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the selected learner need
    assert result["learner_need"] == "recommend"


def test_determine_learner_need_default():
    """
    Test that unclear messages fall back
    to the default recommend action.
    """

    # Create a message that does not match any known intent
    state = {
        "user_id": 2,
        "user_message": "Hello"
    }

    # Run the decision function
    result = determine_learner_need(state)

    # Verify the default behavior
    assert result["learner_need"] == "recommend"


def test_plan_next_topic():
    """
    Test that the agent selects the next available topic
    from the learner's active learning path.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Simulate the TutorState after the active
        # learning path has already been loaded.
        state = {
            "user_id": 2,
            "learning_path": {
                "learning_path_id": 1,
                "name": "Agentic AI Learning Path",
                "status": "active",
            },
        }

        # Run the planning function
        result = plan_next_topic(
            state=state,
            db=db,
        )

        # Make sure current_topic was added
        assert "current_topic" in result

        # Python and Machine Learning are completed,
        # so Agentic AI should be selected next.
        assert result["current_topic"]["topic_id"] == 3
        assert result["current_topic"]["name"] == "Introduction to Agentic AI"

        # Verify its position and current status
        assert result["current_topic"]["position"] == 3
        assert result["current_topic"]["status"] == "pending"

    finally:
        # Always close the database session after the test
        db.close()



def test_load_topic_mastery():
    """
    Test that the learner's topic mastery
    is loaded into TutorState correctly.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Simulate TutorState after the current topic
        # has already been selected.
        state = {
            "user_id": 2,
            "current_topic": {
                "topic_id": 3,
                "name": "Introduction to Agentic AI",
            },
        }

        # Load the mastery information
        result = load_topic_mastery(
            state=state,
            db=db,
        )

        # Make sure topic_mastery was added
        assert "topic_mastery" in result

        # Verify the mastery belongs to the correct topic
        assert result["topic_mastery"]["topic_id"] == 3

        # Verify the stored test mastery score
        assert result["topic_mastery"]["mastery_score"] == 30

        # Verify the stored weak areas
        assert result["topic_mastery"]["weak_areas"] == [
            "agent planning",
            "tool use",
        ]

    finally:
        # Always close the database session
        db.close()



def test_load_topic_mastery_without_current_topic():
    """
    Test that an empty mastery dictionary is returned
    when no current topic exists in TutorState.
    """

    # Create a real database session
    db = SessionLocal()

    try:
        # Simulate a state before a topic is selected
        state = {
            "user_id": 2,
        }

        # Try to load mastery information
        result = load_topic_mastery(
            state=state,
            db=db,
        )

        # No topic means no mastery information
        assert result == {
            "topic_mastery": {}
        }

    finally:
        # Always close the database session
        db.close()


def test_load_conversation_history():
    """
    Test that recent chat messages are loaded
    into TutorState as conversation history.
    """

    with SessionLocal() as db:
        session = None

        try:
            # Create a temporary chat session
            # for the existing test learner.
            session = ChatSession(
                user_id=2,
                session_name="Agent Memory Test",
                started_at=datetime.now(timezone.utc)
            )

            db.add(session)
            db.flush()

            # Add temporary conversation messages.
            user_message = ChatMessage(
                session_id=session.chat_session_id,
                role="user",
                content="Explain AI agents."
            )

            assistant_message = ChatMessage(
                session_id=session.chat_session_id,
                role="assistant",
                content="An AI agent can make decisions and take actions.",
                agent_action="explain"
            )

            db.add_all([
                user_message,
                assistant_message
            ])
            db.commit()

            # Create the TutorState with the
            # temporary chat session ID.
            state = {
                "user_id": 2,
                "session_id": session.chat_session_id,
                "user_message": "Can you explain it again?"
            }

            # Load conversation memory into the state.
            result = load_conversation_history(
                state=state,
                db=db
            )

            # Verify that conversation history
            # was added to the returned state update.
            assert "conversation_history" in result

            # Verify that both messages were loaded.
            assert len(result["conversation_history"]) == 2

            # Verify chronological message order.
            assert (
                result["conversation_history"][0]["role"]
                == "user"
            )

            assert (
                result["conversation_history"][0]["content"]
                == "Explain AI agents."
            )

            assert (
                result["conversation_history"][1]["role"]
                == "assistant"
            )

        finally:
            # Remove temporary messages first
            # because they reference the chat session.
            if session is not None:
                db.query(ChatMessage).filter(
                    ChatMessage.session_id
                    == session.chat_session_id
                ).delete()

                # Remove the temporary chat session.
                db.query(ChatSession).filter(
                    ChatSession.chat_session_id
                    == session.chat_session_id
                ).delete()

                db.commit()

def test_recommend_next_action_no_mastery():
    """
    Test that the agent recommends explanation
    when no mastery record exists.
    """

    # Simulate a learner with no mastery data yet
    state = {
        "topic_mastery": {}
    }

    # Get the recommended learning action
    result = recommend_next_action(state)

    # A new learner should start with explanation
    assert result["recommended_action"] == "explain"


def test_recommend_next_action_low_mastery():
    """
    Test that low mastery leads to explanation.
    """

    # Simulate low mastery
    state = {
        "topic_mastery": {
            "mastery_score": 30
        }
    }

    # Get the recommended learning action
    result = recommend_next_action(state)

    # Low mastery should require explanation
    assert result["recommended_action"] == "explain"


def test_recommend_next_action_medium_mastery():
    """
    Test that medium mastery leads to practice.
    """

    # Simulate medium mastery
    state = {
        "topic_mastery": {
            "mastery_score": 55
        }
    }

    # Get the recommended learning action
    result = recommend_next_action(state)

    # Medium mastery should require practice
    assert result["recommended_action"] == "practice"


def test_recommend_next_action_high_mastery():
    """
    Test that higher mastery leads to review.
    """

    # Simulate higher mastery
    state = {
        "topic_mastery": {
            "mastery_score": 75
        }
    }

    # Get the recommended learning action
    result = recommend_next_action(state)

    # Higher mastery should lead to review
    assert result["recommended_action"] == "review"


def test_recommend_next_action_strong_mastery():
    """
    Test that strong mastery leads to assessment.
    """

    # Simulate strong mastery
    state = {
        "topic_mastery": {
            "mastery_score": 90
        }
    }

    # Get the recommended learning action
    result = recommend_next_action(state)

    # A high mastery score means the learner
    # has completed the current topic and
    # should move to the next recommended step.
    assert result["recommended_action"] == "recommend"



def test_resolve_action_explicit_explain():
    """
    Test that an explicit learner request
    overrides the mastery-based recommendation.
    """

    # Simulate a learner who explicitly asks for explanation,
    # while the mastery logic recommends practice.
    state = {
        "learner_need": "explain",
        "recommended_action": "practice",
    }

    # Resolve the final agent action
    result = resolve_action(state)

    # The learner's explicit request should be respected
    assert result["recommended_action"] == "explain"


def test_resolve_action_explicit_assess():
    """
    Test that an explicit assessment request
    is used as the final action.
    """

    # Simulate a learner who explicitly asks
    # for an assessment.
    state = {
        "learner_need": "assess",
        "recommended_action": "explain",
    }

    # Resolve the final agent action.
    result = resolve_action(state)

    # The learner explicitly requested an assessment,
    # so the agent should respect that request.
    assert result["recommended_action"] == "assess"


def test_resolve_action_explicit_practice():
    """
    Test that an explicit practice request
    is used as the final action.
    """

    # Simulate a learner who explicitly asks for practice
    state = {
        "learner_need": "practice",
        "recommended_action": "review",
    }

    # Resolve the final agent action
    result = resolve_action(state)

    # The explicit learner request should be used
    assert result["recommended_action"] == "practice"


def test_resolve_action_explicit_review():
    """
    Test that an explicit review request
    is used as the final action.
    """

    # Simulate a learner who explicitly asks for review
    state = {
        "learner_need": "review",
        "recommended_action": "assess",
    }

    # Resolve the final agent action
    result = resolve_action(state)

    # The explicit learner request should be used
    assert result["recommended_action"] == "review"


def test_resolve_action_recommendation():
    """
    Asking for a recommendation should preserve the
    suggested action without executing it immediately.
    """

    state = {
        "learner_need": "recommend",
        "recommended_action": "practice",
    }

    result = resolve_action(state)

    assert result["recommended_action"] == "recommend"
    assert result["suggested_action"] == "practice"


def test_resolve_action_default():
    """
    Test that the default recommendation is used
    when no learner need is available.
    """

    # Simulate a state with only a mastery-based recommendation
    state = {
        "recommended_action": "explain",
    }

    # Resolve the final agent action
    result = resolve_action(state)

    # The existing recommendation should be preserved
    assert result["recommended_action"] == "explain"



def test_tutor_graph_end_to_end(monkeypatch):
    """
    Test the Tutor Agent workflow from START to END
    when the learner asks for a recommendation.
    """

    # Mock topic context retrieval so the graph test
    # does not call the real embedding API.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )
    )

    # Create a fake explanation tool so the graph test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            return (
                "An AI agent is a system that can "
                "reason and take actions toward a goal."
            )

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    # Build the compiled LangGraph workflow.
    graph = build_tutor_graph()

    # The learner asks the Tutor to recommend
    # what they should do next.
    initial_state = {
        "user_id": 2,
        "user_message": "What should I learn next?"
    }

    # Run the complete Tutor Agent workflow.
    result = graph.invoke(initial_state)

    # Verify that conversation history
    # was initialized correctly.
    assert "conversation_history" in result
    assert result["conversation_history"] == []

    # Verify that learner context was loaded.
    assert result["learner_context"]["user_id"] == 2
    assert (
        result["learner_context"]["goal"]
        == "Learn Agentic AI"
    )

    # Verify that the active learning path was loaded.
    assert (
        result["learning_path"]["learning_path_id"]
        == 1
    )
    assert (
        result["learning_path"]["status"]
        == "active"
    )

    # The learner explicitly asked for
    # a recommendation.
    assert result["learner_need"] == "recommend"

    # Agentic AI should be the current topic.
    assert result["current_topic"]["topic_id"] == 3
    assert (
        result["current_topic"]["name"]
        == "Introduction to Agentic AI"
    )

    # Verify the learner's stored mastery.
    assert result["topic_mastery"]["topic_id"] == 3
    assert (
        result["topic_mastery"]["mastery_score"]
        == 30
    )

    # Low mastery means explanation is the
    # suggested learning action.
    assert result["suggested_action"] == "explain"

    # Because the learner asked what to do next,
    # the Tutor should present the recommendation
    # instead of executing it immediately.
    assert (
        result["recommended_action"]
        == "recommend"
    )

    assert result["last_action"] == "recommend"

    assert (
        "Recommended next action:"
        in result["response"]
    )

    assert (
        "Would you like me"
        in result["response"]
    )

def test_tutor_graph_loads_conversation_history(monkeypatch):
    """
    Test that the complete Tutor Agent workflow
    loads recent chat messages into TutorState.
    """
    # Mock topic context retrieval so the graph test
    # does not call the real embedding API.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )
    )

    # Create a fake explanation tool so the graph test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            return (
                "An AI agent is a system that can "
                "reason and take actions toward a goal."
            )

    # Replace the real explanation tool
    # during the graph test.
    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    with SessionLocal() as db:
        session = None

        try:
            # Create a temporary chat session
            # for the existing test learner.
            session = ChatSession(
                user_id=2,
                learning_path_id=1,
                topic_id=3,
                session_name="Graph Memory Test",
                started_at=datetime.now(timezone.utc)
            )

            db.add(session)
            db.flush()

            # Store previous conversation messages
            # for the temporary chat session.
            first_message = ChatMessage(
                session_id=session.chat_session_id,
                role="user",
                content="What is an AI agent?"
            )

            second_message = ChatMessage(
                session_id=session.chat_session_id,
                role="assistant",
                content="An AI agent can make decisions and take actions.",
                agent_action="explain"
            )

            db.add_all([
                first_message,
                second_message
            ])

            db.commit()

            # Build the complete Tutor Agent workflow.
            graph = build_tutor_graph()

            # Run the graph using the temporary
            # chat session created for this test.
            initial_state = {
                "user_id": 2,
                "session_id": session.chat_session_id,
                "user_message": "What should I learn next?"
            }

            result = graph.invoke(initial_state)

            # Verify that conversation history
            # was added to TutorState.
            assert "conversation_history" in result

            # Verify that both previous messages
            # were loaded from the database.
            assert len(result["conversation_history"]) == 2

            # Verify that messages remain
            # in chronological conversation order.
            assert (
                result["conversation_history"][0]["role"]
                == "user"
            )

            assert (
                result["conversation_history"][0]["content"]
                == "What is an AI agent?"
            )

            assert (
                result["conversation_history"][1]["role"]
                == "assistant"
            )

            assert (
                result["conversation_history"][1]["content"]
                == (
                    "An AI agent can make decisions "
                    "and take actions."
                )
            )

        finally:
            # Roll back any failed transaction
            # before cleaning up test data.
            db.rollback()

            if session is not None:
                # Delete temporary messages first
                # because they reference the session.
                db.query(ChatMessage).filter(
                    ChatMessage.session_id
                    == session.chat_session_id
                ).delete()

                # Delete the temporary chat session.
                db.query(ChatSession).filter(
                    ChatSession.chat_session_id
                    == session.chat_session_id
                ).delete()

                db.commit()


def test_route_recommended_action_to_teach():
    """
    Test routing an explanation recommendation
    to the teaching node.
    """

    state = {
        "recommended_action": "explain"
    }

    result = route_recommended_action(state)

    assert result == "teach"


def test_route_recommended_action_to_assess():
    """
    Test routing an assessment recommendation
    to the assessment node.
    """

    state = {
        "recommended_action": "assess"
    }

    result = route_recommended_action(state)

    assert result == "assess"


def test_route_recommended_action_to_practice():
    """
    Test routing a practice recommendation
    to the practice node.
    """

    state = {
        "recommended_action": "practice"
    }

    result = route_recommended_action(state)

    assert result == "practice"


def test_route_recommended_action_to_review():
    """
    Test routing a review recommendation
    to the review node.
    """

    state = {
        "recommended_action": "review"
    }

    result = route_recommended_action(state)

    assert result == "review"


def test_tutor_graph_routes_to_teach(monkeypatch):
    """
    Test that the complete Tutor Agent workflow
    routes the learner to the teaching node.
    """

    # Mock topic context retrieval so the graph test
    # does not call the real embedding API.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )
    )

    # Create a fake explanation tool so the graph test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            return (
                "An AI agent is a system that can "
                "reason and take actions toward a goal."
            )

    # Replace the real explanation tool
    # during the graph test.
    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    # Build the compiled Tutor Agent workflow.
    graph = build_tutor_graph()

    # Create the initial TutorState.
    initial_state = {
        "user_id": 2,
        "user_message": "What should I learn next?"
    }

    # Run the complete workflow.
    result = graph.invoke(initial_state)


def test_route_recommended_action_to_recommend():
    """
    Test that the router sends a completed topic
    to the recommendation workflow.
    """

    # Simulate a final agent decision where
    # the learner has completed the current topic
    # and needs a recommendation for the next step.
    state = {
        "recommended_action": "recommend"
    }

    # Route the final recommended action.
    result = route_recommended_action(state)

    # The workflow should continue
    # to the recommendation node.
    assert result == "recommend"

def test_route_recommendation_request_to_display_node():
    """
    A recommendation question should route to the
    display-only recommendation node.
    """

    state = {
        "learner_need": "recommend",
        "recommended_action": "recommend",
        "suggested_action": "practice",
    }

    result = route_recommended_action(state)

    assert result == "recommend_action"


def test_tutor_graph_routes_high_mastery_to_recommend():
    """
    Test that high mastery recommends progression
    while waiting for learner confirmation before
    updating the learning path.
    """

    # Open a database session for the test.
    with SessionLocal() as db:

        # Load the existing mastery record
        # for the Agentic AI topic.
        mastery = (
            db.query(TopicMastery)
            .filter(
                TopicMastery.user_id == 2,
                TopicMastery.topic_id == 3
            )
            .first()
        )

        # Load the current learning path item.
        path_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == 1,
                LearningPathItem.topic_id == 3
            )
            .first()
        )

        # Load the active learning path.
        learning_path = (
            db.query(LearningPath)
            .filter(
                LearningPath.learning_path_id == 1
            )
            .first()
        )

        # Make sure all required test data exists.
        assert mastery is not None
        assert path_item is not None
        assert learning_path is not None

        # Store original values so the test
        # can restore the database afterward.
        original_score = mastery.mastery_score
        original_status = path_item.status
        original_action = (
            path_item.recommended_action
        )
        original_path_status = (
            learning_path.status
        )

        try:
            # Make sure the learning path is active.
            learning_path.status = "active"

            # Make sure Agentic AI is still pending
            # before running the graph.
            path_item.status = "pending"
            path_item.recommended_action = "explain"

            # Simulate strong mastery.
            mastery.mastery_score = 90

            db.commit()

            # Build the Tutor Agent graph.
            graph = build_tutor_graph()

            # Ask the Tutor what should happen next.
            result = graph.invoke(
                {
                    "user_id": 2,
                    "session_id": 1,
                    "user_message": (
                        "What should I learn next?"
                    )
                }
            )

            # High mastery should suggest progression.
            assert (
                result["suggested_action"]
                == "recommend"
            )

            # The Tutor should display the recommendation
            # and wait for learner confirmation.
            assert (
                result["recommended_action"]
                == "recommend"
            )

            assert (
                result["last_action"]
                == "recommend"
            )

            assert (
                "Recommended next action:"
                in result["response"]
            )

            assert (
                "Continue to the next topic"
                in result["response"]
            )

            assert (
                "Would you like to continue "
                "to the next topic?"
                in result["response"]
            )

            # Refresh database state because the graph
            # uses separate database sessions.
            db.expire_all()

            updated_item = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id
                    == 1,
                    LearningPathItem.topic_id == 3
                )
                .first()
            )

            assert updated_item is not None

            # No confirmation has been given yet,
            # so the topic must remain pending.
            assert (
                updated_item.status
                == "pending"
            )

            # The database value should remain unchanged
            # until progression is confirmed.
            assert (
                updated_item.recommended_action
                == "explain"
            )

            updated_path = (
                db.query(LearningPath)
                .filter(
                    LearningPath.learning_path_id
                    == 1
                )
                .first()
            )

            assert updated_path is not None

            # The full path must also remain active
            # until the learner confirms progression.
            assert (
                updated_path.status
                == "active"
            )

        finally:
            # Roll back any failed transaction
            # before restoring the test data.
            db.rollback()

            mastery = (
                db.query(TopicMastery)
                .filter(
                    TopicMastery.user_id == 2,
                    TopicMastery.topic_id == 3
                )
                .first()
            )

            path_item = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id
                    == 1,
                    LearningPathItem.topic_id == 3
                )
                .first()
            )

            learning_path = (
                db.query(LearningPath)
                .filter(
                    LearningPath.learning_path_id
                    == 1
                )
                .first()
            )

            # Restore the original mastery score.
            if mastery is not None:
                mastery.mastery_score = (
                    original_score
                )

            # Restore the original path-item state.
            if path_item is not None:
                path_item.status = (
                    original_status
                )
                path_item.recommended_action = (
                    original_action
                )

            # Restore the original path status.
            if learning_path is not None:
                learning_path.status = (
                    original_path_status
                )

            db.commit()


def test_update_learning_path_node_after_high_mastery():
    """
    Test that the graph node marks the current topic
    as completed after a high mastery score.
    """

    # Open a database session for test setup.
    with SessionLocal() as db:

        # Load the existing Agentic AI mastery record.
        mastery = (
            db.query(TopicMastery)
            .filter(
                TopicMastery.user_id == 2,
                TopicMastery.topic_id == 3
            )
            .first()
        )

        # Load the current Agentic AI learning path item.
        path_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == 1,
                LearningPathItem.topic_id == 3
            )
            .first()
        )

        # Make sure the required test data exists.
        assert mastery is not None
        assert path_item is not None

        # Store the original values so the database
        # can be restored after the test.
        original_score = mastery.mastery_score
        original_status = path_item.status
        original_action = path_item.recommended_action

        try:
            # Simulate a successful assessment.
            mastery.mastery_score = 90

            # Keep the topic pending before running
            # the update node.
            path_item.status = "pending"
            path_item.recommended_action = "explain"

            db.commit()

            # Create the state expected by
            # the update learning path node.
            state = {
                "user_id": 2,
                "learning_path": {
                    "learning_path_id": 1
                },
                "current_topic": {
                    "topic_id": 3,
                    "name": "Introduction to Agentic AI"
                }
            }

            # Run the graph node.
            result = update_learning_path_node(
                state
            )

            # Verify the state returned by the node.
            assert (
                result["current_topic"]["status"]
                == "completed"
            )

            assert (
                result["current_topic"][
                    "recommended_action"
                ]
                == "recommend"
            )

            # Refresh the database objects because
            # the graph node used a separate session.
            db.expire_all()

            updated_item = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id
                    == 1,
                    LearningPathItem.topic_id == 3
                )
                .first()
            )

            # Verify that the update was persisted
            # in the database.
            assert updated_item.status == "completed"

            assert (
                updated_item.recommended_action
                == "recommend"
            )

        finally:
            # Restore the original test data.
            mastery = (
                db.query(TopicMastery)
                .filter(
                    TopicMastery.user_id == 2,
                    TopicMastery.topic_id == 3
                )
                .first()
            )

            path_item = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id
                    == 1,
                    LearningPathItem.topic_id == 3
                )
                .first()
            )

            mastery.mastery_score = original_score
            path_item.status = original_status
            path_item.recommended_action = (
                original_action
            )

            db.commit()

def test_recommend_action_node_displays_reason_without_execution():
    """
    The recommendation display node should explain the
    suggested action without generating practice or
    updating the learning path.
    """

    state = {
        "suggested_action": "practice",
        "recommendation_reason": (
            "Mastery is 60%. The learner understands "
            "some of the topic but needs additional practice."
        ),
        "topic_mastery": {
            "mastery_score": 60,
            "weak_areas": [
                {
                    "area": "Using the break statement",
                    "reason": "Incorrect break condition.",
                },
                {
                    "area": "Calculating an average",
                    "reason": "Incorrect average calculation.",
                },
            ],
        },
    }

    result = recommend_action_node(state)

    assert "Recommended next action" in result["response"]
    assert "Practice" in result["response"]
    assert "Mastery is 60%" in result["response"]
    assert "Using the break statement" in result["response"]
    assert "Calculating an average" in result["response"]
    assert "Would you like me to start" in result["response"]
    assert result["last_action"] == "recommend"

    assert "current_topic" not in result
    assert "next_topic_id" not in result


def test_recommend_node_selects_next_topic():
    """
    Test that the recommendation node selects
    the next available topic in the learning path.
    """

    # Open a database session for test setup.
    with SessionLocal() as db:

        # Load the Machine Learning item.
        ml_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == 1,
                LearningPathItem.topic_id == 2
            )
            .first()
        )

        # Load the Agentic AI item.
        agentic_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == 1,
                LearningPathItem.topic_id == 3
            )
            .first()
        )

        # Make sure the required test data exists.
        assert ml_item is not None
        assert agentic_item is not None

        # Store the original values so the database
        # can be restored after the test.
        original_ml_status = ml_item.status
        original_agentic_status = agentic_item.status

        try:
            # Simulate a state where Machine Learning
            # has been completed and Agentic AI is next.
            ml_item.status = "completed"
            agentic_item.status = "pending"

            db.commit()

            # Create the state expected
            # by the recommendation node.
            state = {
                "user_id": 2,
                "learning_path": {
                    "learning_path_id": 1
                },
                "current_topic": {
                    "topic_id": 2,
                    "name": "Machine Learning Basics"
                }
            }

            # Run the recommendation node.
            result = recommend_node(state)

            # Verify that Agentic AI was selected
            # as the learner's next topic.
            assert (
                result["next_topic_id"]
                == 3
            )

            assert (
                result["current_topic"]["topic_id"]
                == 3
            )

            assert (
                result["current_topic"]["name"]
                == "Introduction to Agentic AI"
            )

        finally:
            # Restore the original learning path
            # item statuses after the test.
            ml_item.status = original_ml_status
            agentic_item.status = (
                original_agentic_status
            )

            db.commit()


def test_build_recommendation_without_assessment():
    """
    The learner should receive an explanation recommendation
    when no assessment result is available.
    """

    result = build_recommendation(
        mastery_score=None
    )

    assert result["recommended_action"] == "explain"
    assert "No assessment result" in result["recommendation_reason"]


def test_build_recommendation_low_mastery():
    """
    Low mastery should recommend additional explanation.
    """

    result = build_recommendation(
        mastery_score=30
    )

    assert result["recommended_action"] == "explain"
    assert "30%" in result["recommendation_reason"]


def test_build_recommendation_moderate_mastery():
    """
    Moderate mastery should recommend practice.
    """

    result = build_recommendation(
        mastery_score=55
    )

    assert result["recommended_action"] == "practice"
    assert "55%" in result["recommendation_reason"]


def test_build_recommendation_good_mastery():
    """
    Good mastery should recommend review.
    """

    result = build_recommendation(
        mastery_score=75
    )

    assert result["recommended_action"] == "review"
    assert "75%" in result["recommendation_reason"]


def test_build_recommendation_high_mastery():
    """
    High mastery without weak areas should allow
    the learner to progress.
    """

    result = build_recommendation(
        mastery_score=90
    )

    assert result["recommended_action"] == "recommend"
    assert "ready for the next learning step" in (
        result["recommendation_reason"]
    )


def test_build_recommendation_high_mastery_with_weak_areas():
    """
    High mastery should still allow progression when minor
    weak areas remain, while mentioning them in the reason.
    """

    result = build_recommendation(
        mastery_score=90,
        weak_areas=["tool error handling"]
    )

    assert result["recommended_action"] == "recommend"
    assert "weak areas" in result["recommendation_reason"]




def test_recommend_node_builds_personalized_response(monkeypatch):
    """
    The recommendation node should select the next eligible
    topic and include the recommendation reason and learner goal.
    """

    # Mock the next eligible topic so the test does not
    # depend on real database learning path data.
    next_topic = {
        "topic_id": 4,
        "name": "Building Your First Agent",
        "description": "Introduction to building AI agents.",
        "difficulty_level": "intermediate",
        "position": 4,
        "status": "pending",
        "recommended_action": "explain",
    }

    monkeypatch.setattr(
        "app.agent.graph.select_next_topic",
        lambda db, learning_path_id: next_topic
    )

    state = {
        "user_id": 2,
        "learning_path": {
            "learning_path_id": 1,
            "goal": "Learn Agentic AI",
            "status": "active",
        },
        "learner_context": {
            "goal": "Learn Agentic AI",
        },
        "recommendation_reason": (
            "Mastery is 90%. The learner has demonstrated "
            "strong mastery and is ready for the next learning step."
        ),
    }

    result = recommend_node(state)

    assert result["next_topic_id"] == 4
    assert result["current_topic"] == next_topic

    assert "Building Your First Agent" in result["response"]
    assert "completed this topic" in result["response"]
    assert "ready to continue" in result["response"]


def test_recommend_node_completes_path_when_no_topic_remains(
    monkeypatch
):
    """
    The recommendation node should complete the learning path
    when no eligible incomplete topics remain.
    """

    # Simulate a learning path with no remaining topics.
    monkeypatch.setattr(
        "app.agent.graph.select_next_topic",
        lambda db, learning_path_id: {}
    )

    # Mock path completion so the test does not
    # modify the real database.
    monkeypatch.setattr(
        "app.agent.graph.complete_learning_path",
        lambda db, learning_path_id: {
            "learning_path_id": learning_path_id,
            "status": "completed",
        }
    )

    state = {
        "user_id": 2,
        "learning_path": {
            "learning_path_id": 1,
            "goal": "Learn Agentic AI",
            "status": "active",
        },
    }

    result = recommend_node(state)

    assert result["learning_path"]["status"] == "completed"
    assert "completed all topics" in result["response"]



def test_prepare_assessment_inputs():
    """
    Assessment inputs should include the current topic,
    learner level, assessment type, and user request.
    """

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "initial_level": "beginner",
            "current_level": "intermediate",
        },
        "assessment_type": "topic",
        "user_message": (
            "Test me on tool usage in AI agents."
        ),
    }

    result = prepare_assessment_inputs(state)

    assert result["topic_id"] == 12
    assert result["topic_name"] == "Building Your First Agent"
    assert result["student_level"] == "intermediate"
    assert result["assessment_type"] == "topic"
    assert result["user_message"] == (
        "Test me on tool usage in AI agents."
    )


    assert result["student_level"] == "intermediate"
    assert result["assessment_type"] == "topic"

    assert (
        result["user_message"]
        == "Test me on tool usage in AI agents."
    )


def test_prepare_assessment_inputs_without_topic():
    """
    Assessment preparation should stop safely
    when no current topic is available.
    """

    state = {
        "current_topic": {},
        "learner_context": {
            "current_level": "intermediate",
        },
        "user_message": (
            "Test me on tool usage in AI agents."
        ),
    }

    result = prepare_assessment_inputs(state)

    assert result == {}


def test_generate_assessment_node_generates_quiz(
    monkeypatch
):
    """
    The assessment node should use the learner's
    request for RAG retrieval, generate a quiz,
    and store the generated questions in TutorState.
    """

    retrieved_query = {}

    # Mock RAG retrieval and capture its inputs
    # so the semantic retrieval query can be verified.
    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        retrieved_query["topic_id"] = topic_id
        retrieved_query["topic_name"] = topic_name
        retrieved_query["retrieval_query"] = (
            retrieval_query
        )

        return (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    # Create a fake quiz response matching the
    # structure returned by the quiz generation tool.
    fake_quiz = {
        "questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "question_type": "multiple_choice",
                "difficulty": "intermediate",
                "options": [
                    "A",
                    "B",
                    "C",
                    "D",
                ],
                "correct_answer": "A",
            }
        ]
    }

    # Mock the quiz tool so no LLM API
    # call is made during the test.
    class FakeQuizTool:
        def invoke(self, inputs):
            return json.dumps(fake_quiz)

    monkeypatch.setattr(
        "app.agent.graph.generate_quiz",
        FakeQuizTool()
    )

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "assessment_type": "topic",
        "user_message": (
            "Test me on tool usage in AI agents."
        ),
    }

    result = generate_assessment_node(state)

    # Verify that retrieval remains restricted
    # to the current topic.
    assert retrieved_query["topic_id"] == 12

    assert (
        retrieved_query["topic_name"]
        == "Building Your First Agent"
    )

    # Verify that the learner's actual request
    # is used as the semantic retrieval query.
    assert (
        retrieved_query["retrieval_query"]
        == "Test me on tool usage in AI agents."
    )

    assert result["assessment_type"] == "topic"
    assert len(result["assessment_questions"]) == 1

    assert (
        result["assessment_questions"][0]["topic_id"]
        == 12
    )

    assert "questions" in result["response"]


def test_generate_assessment_node_stops_without_rag_context(
    monkeypatch
):
    """
    The assessment node should stop safely when
    no learning context is available from RAG.
    """

    # Accept retrieval_query because the assessment
    # node now passes the learner's request to RAG.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id,
        topic_name,
        retrieval_query=None: ""
    )

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "assessment_type": "topic",
        "user_message": (
            "Test me on tool usage in AI agents."
        ),
    }

    result = generate_assessment_node(state)

    assert result.get("assessment_questions") is None
    assert "no learning context" in result["response"]


def test_prepare_assessment_responses():
    """
    Generated questions should be combined with
    the learner's submitted answers.
    """

    state = {
        "assessment_questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            }
        ],
    }

    result = prepare_assessment_responses(state)

    assert len(result) == 1
    assert result[0]["topic_id"] == 12
    assert result[0]["learner_answer"] == "A"
    assert result[0]["correct_answer"] == "A"


def test_prepare_assessment_responses_missing_answers():
    """
    Assessment response preparation should stop safely
    when learner answers are missing.
    """

    state = {
        "assessment_questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "correct_answer": "A",
            }
        ],
        "assessment_answers": [],
    }

    result = prepare_assessment_responses(state)

    assert result == []


def test_evaluate_assessment_responses(
    monkeypatch
):
    """
    Assessment responses should be evaluated
    without calling the real LLM during testing.
    """

    responses = [
        {
            "topic_id": 12,
            "question_text": "What is an AI agent?",
            "question_type": "multiple_choice",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "learner_answer": "A",
        }
    ]

    fake_evaluation = {
        "is_correct": True,
        "score_awarded": 1,
        "feedback": "Correct answer.",
    }

    class FakeEvaluationTool:
        """
        Fake evaluation tool used to avoid
        calling the real LLM during testing.
        """

        def invoke(self, inputs):
            return json.dumps(
                fake_evaluation
            )

    monkeypatch.setattr(
        "app.agent.graph.evaluate_answer",
        FakeEvaluationTool()
    )

    result = evaluate_assessment_responses(
        responses
    )

    assert len(result) == 1
    assert result[0]["is_correct"] is True
    assert result[0]["score_awarded"] == 1
    assert result[0]["feedback"] == "Correct answer."
    assert result[0]["learner_answer"] == "A"


def test_evaluate_assessment_responses_empty():
    """
    Assessment evaluation should stop safely
    when no responses are available.
    """

    result = evaluate_assessment_responses([])

    assert result == []


def test_submit_assessment_node(monkeypatch):
    """
    The assessment submission node should evaluate
    learner responses, save the result, reload the
    updated topic mastery, synchronize the learning
    path item, and return the completed result.
    """

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "assessment_type": "topic",
        "assessment_questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            }
        ],
    }

    # Mock the evaluation step so the test
    # does not call the real LLM.
    evaluated_responses = [
        {
            "topic_id": 12,
            "question_text": "What is an AI agent?",
            "question_type": "multiple_choice",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "learner_answer": "A",
            "is_correct": True,
            "score_awarded": 1,
            "feedback": "Correct answer.",
        }
    ]

    monkeypatch.setattr(
        "app.agent.graph.evaluate_assessment_responses",
        lambda responses: evaluated_responses
    )

    # Create a fake assessment attempt matching
    # the fields returned by the database service.
    class FakeAssessmentAttempt:
        assessment_attempt_id = 100
        score = 1
        max_score = 1

    # Mock database persistence so this test
    # does not modify PostgreSQL.
    monkeypatch.setattr(
        "app.agent.graph.save_assessment_result",
        lambda db,
        user_id,
        topic_id,
        assessment_type,
        questions: FakeAssessmentAttempt()
    )

    # Mock the updated mastery that would normally
    # be loaded from PostgreSQL after submission.
    updated_mastery = {
        "mastery_score": 100.0,
        "weak_areas": [],
        "last_assessed_at": None,
    }

    monkeypatch.setattr(
        "app.agent.graph.get_topic_mastery",
        lambda db, user_id, topic_id: updated_mastery
    )

    # Mock the learner's active learning path.
    monkeypatch.setattr(
        "app.agent.graph.get_active_learning_path",
        lambda db, user_id: {
            "learning_path_id": 50,
            "user_id": user_id,
            "name": "Agentic AI Learning Path",
            "status": "active",
        }
    )

    # Simulate synchronization of the current
    # learning path item after high mastery.
    monkeypatch.setattr(
        "app.agent.graph.update_learning_path",
        lambda db,
        user_id,
        learning_path_id,
        topic_id: {
            "learning_path_item_id": 200,
            "learning_path_id": learning_path_id,
            "topic_id": topic_id,
            "status": "completed",
            "recommended_action": "recommend",
        }
    )

    # Run the assessment submission node.
    result = submit_assessment_node(state)

    # Verify the saved assessment result.
    assert result["assessment_result"] == {
        "assessment_attempt_id": 100,
        "score": 1,
        "max_score": 1,
    }

    # Verify that the updated topic mastery
    # is returned to TutorState.
    assert result["topic_mastery"] == updated_mastery

    assert (
        result["topic_mastery"]["mastery_score"]
        == 100.0
    )

    # Verify that the next-step recommendation
    # is calculated from the updated mastery.
    assert result["recommended_action"] == "recommend"

    assert result["recommendation_reason"]

    # Verify that the learning path item was
    # synchronized with the latest mastery.
    assert (
        result["current_topic"]["status"]
        == "completed"
    )

    assert (
        result["current_topic"]["recommended_action"]
        == "recommend"
    )

    # Verify the final response shown
    # after assessment submission.
    assert "Assessment completed" in result["response"]
    assert "1/1" in result["response"]

    # Verify that the learner sees the updated
    # mastery and recommended next step.
    assert "Mastery: 100%" in result["response"]
    assert (
        "Recommended next step: recommend"
        in result["response"]
    )


def test_submit_assessment_node_low_mastery(monkeypatch):
    """
    A low assessment mastery should keep the
    current topic pending and recommend explanation.
    """

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
            "status": "completed",
        },
        "assessment_type": "topic",
        "assessment_questions": [
            {
                "topic_id": 27,
                "question_text": (
                    "What does a = 4 do?"
                ),
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "B",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            }
        ],
    }

    # Simulate an incorrect learner response.
    evaluated_responses = [
        {
            "topic_id": 27,
            "question_text": (
                "What does a = 4 do?"
            ),
            "question_type": "multiple_choice",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "learner_answer": "A",
            "is_correct": False,
            "score_awarded": 0,
            "feedback": "Incorrect answer.",
        }
    ]

    monkeypatch.setattr(
        "app.agent.graph.evaluate_assessment_responses",
        lambda responses: evaluated_responses
    )

    # Create a fake low-scoring assessment attempt.
    class FakeAssessmentAttempt:
        assessment_attempt_id = 201
        score = 0
        max_score = 1

    monkeypatch.setattr(
        "app.agent.graph.save_assessment_result",
        lambda db,
        user_id,
        topic_id,
        assessment_type,
        questions: FakeAssessmentAttempt()
    )

    # Simulate the new mastery produced
    # after the low assessment score.
    updated_mastery = {
        "mastery_score": 20.0,
        "weak_areas": [
            "Variable assignment"
        ],
        "last_assessed_at": None,
    }

    monkeypatch.setattr(
        "app.agent.graph.get_topic_mastery",
        lambda db, user_id, topic_id: updated_mastery
    )

    # Mock the learner's active learning path.
    monkeypatch.setattr(
        "app.agent.graph.get_active_learning_path",
        lambda db, user_id: {
            "learning_path_id": 50,
            "user_id": user_id,
            "name": "Agentic AI Learning Path",
            "status": "active",
        }
    )

    # Simulate synchronization after mastery falls
    # below the topic completion threshold.
    monkeypatch.setattr(
        "app.agent.graph.update_learning_path",
        lambda db,
        user_id,
        learning_path_id,
        topic_id: {
            "learning_path_item_id": 300,
            "learning_path_id": learning_path_id,
            "topic_id": topic_id,
            "status": "pending",
            "recommended_action": "explain",
        }
    )

    result = submit_assessment_node(state)

    # Verify that the low mastery is returned.
    assert (
        result["topic_mastery"]["mastery_score"]
        == 20.0
    )

    # Verify that the learner should remain
    # in the explanation stage.
    assert result["recommended_action"] == "explain"

    # Verify that the current topic is no longer
    # treated as completed.
    assert (
        result["current_topic"]["status"]
        == "pending"
    )

    assert (
        result["current_topic"]["recommended_action"]
        == "explain"
    )

    # Verify the response reflects the new result.
    assert "Assessment completed" in result["response"]
    assert "Mastery: 20%" in result["response"]

    assert (
        "Recommended next step: explain"
        in result["response"]
    )


def test_submit_assessment_node_incomplete_answers():
    """
    Assessment submission should stop safely
    when learner answers are incomplete.
    """

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "assessment_type": "topic",
        "assessment_questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "correct_answer": "A",
            }
        ],
        "assessment_answers": [],
    }

    result = submit_assessment_node(state)

    assert result.get("assessment_result") is None

    assert (
        "answers are incomplete"
        in result["response"]
    )

def test_submit_assessment_updates_path_after_low_mastery(
    monkeypatch
):
    """
    A low assessment mastery should return the current
    learning path item to pending and recommend explanation.
    """

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
            "status": "completed",
        },
        "assessment_type": "topic",
        "assessment_questions": [
            {
                "topic_id": 27,
                "question_text": "What does a = 4 do?",
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "B",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            }
        ],
    }

    evaluated_responses = [
        {
            "topic_id": 27,
            "question_text": "What does a = 4 do?",
            "question_type": "multiple_choice",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "learner_answer": "A",
            "is_correct": False,
            "score_awarded": 0,
            "feedback": "Incorrect answer.",
        }
    ]

    monkeypatch.setattr(
        "app.agent.graph.evaluate_assessment_responses",
        lambda responses: evaluated_responses
    )

    class FakeAssessmentAttempt:
        assessment_attempt_id = 200
        score = 0
        max_score = 1

    monkeypatch.setattr(
        "app.agent.graph.save_assessment_result",
        lambda db,
        user_id,
        topic_id,
        assessment_type,
        questions: FakeAssessmentAttempt()
    )

    updated_mastery = {
        "mastery_score": 20.0,
        "weak_areas": [
            "Variable assignment"
        ],
        "last_assessed_at": None,
    }

    monkeypatch.setattr(
        "app.agent.graph.get_topic_mastery",
        lambda db,
        user_id,
        topic_id: updated_mastery
    )

    monkeypatch.setattr(
        "app.agent.graph.get_active_learning_path",
        lambda db, user_id: {
            "learning_path_id": 50,
            "user_id": user_id,
            "name": "Agentic AI Learning Path",
            "status": "active",
        }
    )

    # Simulate the path synchronization expected
    # after mastery drops below the completion threshold.
    monkeypatch.setattr(
        "app.agent.graph.update_learning_path",
        lambda db,
        user_id,
        learning_path_id,
        topic_id: {
            "learning_path_item_id": 300,
            "learning_path_id": learning_path_id,
            "topic_id": topic_id,
            "status": "pending",
            "recommended_action": "explain",
        }
    )

    result = submit_assessment_node(state)

    assert (
        result["topic_mastery"]["mastery_score"]
        == 20.0
    )

    assert (
        result["recommended_action"]
        == "explain"
    )

    assert (
        result["current_topic"]["status"]
        == "pending"
    )

    assert (
        result["current_topic"]["recommended_action"]
        == "explain"
    )

    assert "Mastery: 20%" in result["response"]


def test_route_assessment_to_generate():
    """
    Assessment routing should generate a new quiz
    when no existing assessment answers are available.
    """

    state = {
        "assessment_questions": [],
        "assessment_answers": [],
    }

    result = route_assessment(state)

    assert result == "generate_assessment"


def test_route_assessment_to_submit():
    """
    Assessment routing should submit the assessment
    when questions and learner answers are available.
    """

    state = {
        "assessment_questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "correct_answer": "A",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            }
        ],
    }

    result = route_assessment(state)

    assert result == "submit_assessment"



def test_prepare_personalized_inputs():
    """
    Teaching inputs should be prepared from
    the current topic, learner context, and user request.
    """

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "initial_level": "beginner",
            "current_level": "intermediate",
        },
        "user_message": (
            "Explain how an AI agent uses tools."
        ),
    }

    result = prepare_personalized_inputs(state)

    assert result["topic_id"] == 12

    assert (
        result["topic_name"]
        == "Building Your First Agent"
    )

    assert (
        result["student_level"]
        == "intermediate"
    )

    assert (
        result["user_message"]
        == "Explain how an AI agent uses tools."
    )

    # No mastery data was provided in this state.
    assert result["mastery_score"] is None

    assert result["weak_areas"] == []

    # No learner preferences were provided.
    assert result["preferred_format"] is None
    assert result["preferred_pace"] is None

    # No chat session history was provided.
    assert result["conversation_history"] == []


def test_prepare_personalized_inputs_without_topic():
    """
    Teaching input preparation should stop safely
    when no current topic is available.
    """

    state = {
        "current_topic": {},
        "learner_context": {
            "current_level": "intermediate",
        },
    }

    result = prepare_personalized_inputs(state)

    assert result == {}


def test_teach_node_generates_explanation(
    monkeypatch
):
    """
    The teaching node should use the learner's
    specific request for RAG retrieval and generate
    a personalized explanation.
    """

    retrieved_query = {}

    # Mock RAG retrieval so the test does not
    # require embeddings or a real vector database.
    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        # Capture the retrieval inputs so the test
        # can verify the correct topic and query.
        retrieved_query["topic_id"] = topic_id
        retrieved_query["topic_name"] = topic_name
        retrieved_query["retrieval_query"] = (
            retrieval_query
        )

        return (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    # Create a fake explanation tool so the test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            assert (
                'Teach the topic "Building Your First Agent".'
                in inputs["topic"]
            )

            assert (
                retrieved_query["retrieval_query"]
                == (
                    "Building Your First Agent: "
                    "Explain how an AI agent uses tools."
                )
            )
            assert (
                inputs["student_level"]
                == "intermediate"
            )
            assert (
                "AI agents can reason"
                in inputs["context"]
            )

            return (
                "An AI agent is a system that can "
                "reason and take actions toward a goal."
            )

    # Replace the real explanation tool
    # with the fake tool during this test.
    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "user_message": (
            "Explain how an AI agent uses tools."
        ),
    }

    result = teach_node(state)

    # Verify that retrieval remains restricted
    # to the learner's current topic.
    assert retrieved_query["topic_id"] == 12

    assert (
        retrieved_query["topic_name"]
        == "Building Your First Agent"
    )

    # Verify that the learner's actual request
    # is used as the semantic retrieval query.
    assert (
        "Explain how an AI agent uses tools."
        in retrieved_query["retrieval_query"]
    )

    # Verify the final teaching response.
    assert result["response"] == (
        "An AI agent is a system that can "
        "reason and take actions toward a goal."
    )


def test_teach_node_stops_without_rag_context(
    monkeypatch
):
    """
    The teaching node should stop safely when
    no learning context is available from RAG.
    """

    # Simulate a topic with no available
    # learning material in the knowledge base.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: ""
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
    }

    result = teach_node(state)

    assert "no learning context" in result["response"]

def test_teach_node_uses_personalization_data(
    monkeypatch
):
    """
    The teaching node should include learner mastery,
    weak areas, and preferences in the explanation request.
    """

    # Mock RAG retrieval so the test does not
    # call the real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "Python variables store values and can "
            "be reassigned during program execution."
        )
    )

    captured_inputs = {}

    class FakeExplainTool:
        def invoke(self, inputs):
            # Capture the inputs sent to the explanation tool.
            captured_inputs.update(inputs)

            return (
                "Personalized explanation generated."
            )

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "practical_examples",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 20.0,
            "weak_areas": [
                {
                    "area": "Variable reassignment",
                    "reason": "Incorrect understanding.",
                },
                {
                    "area": "Multiple assignment order",
                    "reason": "Values were reversed.",
                },
            ],
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "I find variables confusing.",
            }
        ],
        "user_message": (
            "Explain Python variables."
        ),
    }

    result = teach_node(state)

    personalized_topic = captured_inputs[
        "topic"
    ]

    assert (
        'Teach the topic '
        '"Python Reference Material".'
        in personalized_topic
    )

    assert (
        "20%"
        in personalized_topic
    )

    assert (
        "Variable reassignment"
        in personalized_topic
    )

    assert (
        "Multiple assignment order"
        in personalized_topic
    )

    assert (
        "practical_examples"
        in personalized_topic
    )

    assert (
        "slow"
        in personalized_topic
    )

    assert (
        "Explain Python variables."
        in personalized_topic
    )

    assert (
        captured_inputs["student_level"]
        == "beginner"
    )

    assert (
        "Python variables store values"
        in captured_inputs["context"]
    )

    assert (
        result["response"]
        == "Personalized explanation generated."
    )



def test_practice_node_generates_personalized_practice(
    monkeypatch
):
    """
    The practice node should use the learner's
    specific request for RAG retrieval and generate
    personalized practice activities.
    """

    retrieved_query = {}

    # Mock RAG retrieval so the test does not
    # require embeddings or a real vector database.
    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        # Capture retrieval inputs so the test can
        # verify the current topic and user request.
        retrieved_query["topic_id"] = topic_id
        retrieved_query["topic_name"] = topic_name
        retrieved_query["retrieval_query"] = (
            retrieval_query
        )

        return (
            "AI agents can use tools to interact "
            "with external systems."
        )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    # Create a fake practice tool so the test
    # does not call the real language model.
    class FakePracticeTool:
        def invoke(self, inputs):
            # Verify that the practice request contains
            # the current topic.
            assert (
                'Practice the topic '
                '"Building Your First Agent".'
                in inputs["topic"]
            )

            # Verify that the learner's mastery score
            # is included in the personalized request.
            assert (
                "55%"
                in inputs["topic"]
            )

            # Verify that the learner's current request
            # is included in the personalized request.
            assert (
                "Give me practice questions "
                "about tool selection."
                in inputs["topic"]
            )

            assert (
                inputs["student_level"]
                == "intermediate"
            )

            assert (
                "AI agents can use tools"
                in inputs["context"]
            )

            # Verify that weak areas are passed
            # to the practice tool as JSON.
            assert json.loads(
                inputs["weak_areas"]
            ) == [
                "Tool selection",
                "Tool arguments",
            ]

            # A learner with 55% mastery in an Agentic AI
            # topic should receive short-answer practice.
            assert (
                inputs["practice_type"]
                == "short_answer"
            )

            assert inputs["num_items"] == 5

            return (
                "Practice activity focused on "
                "tool selection and tool arguments."
            )

    # Replace the real practice generation tool
    # with the fake tool during this test.
    monkeypatch.setattr(
        "app.agent.graph.generate_practice",
        FakePracticeTool()
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "topic_mastery": {
            "mastery_score": 55,
            "weak_areas": [
                "Tool selection",
                "Tool arguments",
            ],
        },
        "user_message": (
            "Give me practice questions about tool selection."
        ),
    }

    result = practice_node(state)

    # Verify that retrieval remains restricted
    # to the learner's current topic.
    assert retrieved_query["topic_id"] == 12

    assert (
        retrieved_query["topic_name"]
        == "Building Your First Agent"
    )
    # Verify that the learner's request and identified
    # weak areas are included in the retrieval query.
    retrieval_query = retrieved_query[
        "retrieval_query"
    ]

    assert (
        "Give me practice questions about tool selection."
        in retrieval_query
    )

    assert "Tool selection" in retrieval_query
    assert "Tool arguments" in retrieval_query

    # Verify the final practice response.
    assert result["response"] == (
        "Practice activity focused on "
        "tool selection and tool arguments."
    )


def test_practice_node_stops_without_rag_context(
    monkeypatch
):
    """
    The practice node should stop safely when
    no learning context is available from RAG.
    """

    # Simulate a topic with no available
    # learning material in the knowledge base.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: ""
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "topic_mastery": {
            "mastery_score": 55,
            "weak_areas": [
                "Tool selection",
            ],
        },
    }

    result = practice_node(state)

    assert "no learning context" in result["response"]



def test_review_node_generates_focused_review(
    monkeypatch
):
    """
    The review node should use the learner's
    specific request for RAG retrieval and generate
    a review focused on weak areas.
    """

    retrieved_query = {}

    # Mock RAG retrieval and capture its inputs
    # so the semantic query can be verified.
    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        retrieved_query["topic_id"] = topic_id
        retrieved_query["topic_name"] = topic_name
        retrieved_query["retrieval_query"] = (
            retrieval_query
        )

        return (
            "AI agents can reason, use tools, "
            "and interact with external systems."
        )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    # Create a fake explanation tool so the test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            # Verify that the review request contains
            # the topic and the learner's weak areas.
            assert (
                "Building Your First Agent"
                in inputs["topic"]
            )
            assert (
                "Tool selection"
                in inputs["topic"]
            )
            assert (
                "Tool arguments"
                in inputs["topic"]
            )

            # Verify learner level and RAG context.
            assert (
                inputs["student_level"]
                == "intermediate"
            )
            assert (
                "AI agents can reason"
                in inputs["context"]
            )

            return (
                "Focused review of tool selection "
                "and tool arguments."
            )

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                "Tool selection",
                "Tool arguments",
            ],
        },
        "user_message": (
            "Review tool selection with me."
        ),
    }

    result = review_node(state)

    # Verify retrieval remains restricted
    # to the learner's current topic.
    assert retrieved_query["topic_id"] == 12

    assert (
        retrieved_query["topic_name"]
        == "Building Your First Agent"
    )

    # Verify the learner's actual request is
    # used as the semantic retrieval query.
    assert (
        retrieved_query["retrieval_query"]
        == "Review tool selection with me."
    )

    assert result["response"] == (
        "Focused review of tool selection "
        "and tool arguments."
    )


def test_review_node_stops_without_rag_context(
    monkeypatch
):
    """
    The review node should stop safely when
    no learning context is available from RAG.
    """

    # Accept retrieval_query because review_node
    # now passes the learner's request to RAG.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id,
        topic_name,
        retrieval_query=None: ""
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                "Tool selection",
            ],
        },
        "user_message": (
            "Review tool selection with me."
        ),
    }

    result = review_node(state)

    assert "no learning context" in result["response"]


def test_review_node_generates_general_review_without_weak_areas(
    monkeypatch
):
    """
    The review node should generate a general topic review
    when no weak areas are available while still using
    the learner's request for RAG retrieval.
    """

    retrieved_query = {}

    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        retrieved_query["retrieval_query"] = (
            retrieval_query
        )

        return (
            "AI agents can reason, use tools, "
            "and interact with external systems."
        )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    # Create a fake explanation tool so the test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            # With no weak areas, the generated review
            # should summarize the topic generally.
            assert (
                "Building Your First Agent"
                in inputs["topic"]
            )
            assert (
                "Summarize the key concepts"
                in inputs["topic"]
            )
            assert (
                "weak areas"
                not in inputs["topic"]
            )

            assert (
                inputs["student_level"]
                == "intermediate"
            )
            assert (
                "AI agents can reason"
                in inputs["context"]
            )

            return (
                "General review of the key concepts "
                "for building an AI agent."
            )

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [],
        },
        "user_message": (
            "Review this topic with me."
        ),
    }

    result = review_node(state)

    assert (
        retrieved_query["retrieval_query"]
        == "Review this topic with me."
    )

    assert result["response"] == (
        "General review of the key concepts "
        "for building an AI agent."
    )


def test_generate_initial_diagnostic_without_selected_path():
    """
    Test that the diagnostic cannot start
    before the learner selects a learning path.
    """

    with SessionLocal() as db:
        state = {
            "learner_context": {
                "current_level": "beginner"
            }
        }

        result = generate_initial_diagnostic_node(
            state,
            db
        )

        assert "select a learning path" in result["response"].lower()
        assert "assessment_questions" not in result


def test_generate_initial_diagnostic_without_required_topics(
    monkeypatch
):
    """
    Test that a learning path is created directly
    when no prerequisite diagnostic is required.
    """

    monkeypatch.setattr(
        "app.agent.graph.create_learning_path",
        lambda db,
        user_id,
        target_topic_id,
        path_name,
        goal: {
            "learning_path_id": 100,
            "user_id": user_id,
            "name": path_name,
            "goal": goal,
            "status": "active",
            "topic_ids": list(range(35, 43)),
        }
    )

    monkeypatch.setattr(
        "app.agent.graph.select_next_topic",
        lambda db, learning_path_id: {
            "topic_id": 35,
            "name": "Ch1 Introduction To Python",
            "status": "pending",
            "recommended_action": "explain",
        }
    )

    with SessionLocal() as db:
        state = {
            "user_id": 2,
            "selected_path": "python",
            "learner_context": {
                "current_level": "beginner"
            },
        }

        result = generate_initial_diagnostic_node(
            state,
            db
        )

    assert result["diagnostic_topics"] == []
    assert result["assessment_questions"] == []

    assert (
        result["learning_path"]["learning_path_id"]
        == 100
    )

    assert result["next_topic_id"] == 35

    assert (
        result["response"]
        == (
            "No prerequisite diagnostic assessment "
            "is required. Your learning path is ready."
        )
    )


def test_generate_initial_diagnostic_agentic_ai(
    monkeypatch
):
    """
    Test that the Agentic AI path generates
    a prerequisite diagnostic across Python
    and Machine Learning.
    """

    quiz_calls = []

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None:
            f"Context for {topic_name}"
    )

    class FakeQuizTool:
        def invoke(self, inputs):
            quiz_calls.append(
                {
                    "topics": inputs["topics"],
                    "num_questions": inputs[
                        "num_questions"
                    ],
                    "assessment_type": inputs[
                        "assessment_type"
                    ],
                }
            )

            topic_id = inputs["topics"][0][
                "topic_id"
            ]

            return json.dumps(
                {
                    "questions": [
                        {
                            "topic_id": topic_id,
                            "question_text": (
                                f"Diagnostic question {i}"
                            ),
                            "question_type": (
                                "multiple_choice"
                            ),
                            "difficulty": "beginner",
                            "options": [
                                "A",
                                "B",
                                "C",
                                "D",
                            ],
                            "correct_answer": "A",
                        }
                        for i in range(
                            1,
                            inputs["num_questions"] + 1
                        )
                    ]
                }
            )

    monkeypatch.setattr(
        "app.agent.graph.generate_quiz",
        FakeQuizTool()
    )

    state = {
        "selected_path": "agentic_ai",
        "learner_context": {
            "current_level": "beginner"
        },
    }

    with SessionLocal() as db:
        python_topic = get_topic_by_key(
            db,
            "python_ch0_1",
        )
        ml_topic = get_topic_by_key(
            db,
            "machine_learning_ch0_1",
        )
        expected_topics = [
            {
                "topic_id": python_topic.topic_id,
                "topic": python_topic.name,
            },
            {
                "topic_id": ml_topic.topic_id,
                "topic": ml_topic.name,
            },
        ]

        result = generate_initial_diagnostic_node(
            state,
            db
        )

    assert result["diagnostic_topics"] == expected_topics

    assert quiz_calls == [
        {
            "topics": [expected_topics[0]],
            "num_questions": 5,
            "assessment_type": "diagnostic",
        },
        {
            "topics": [expected_topics[1]],
            "num_questions": 5,
            "assessment_type": "diagnostic",
        },
    ]

    assert len(
        result["assessment_questions"]
    ) == 10


def test_initial_diagnostic_uses_rag_backed_reference_topics(
    monkeypatch
):
    """
    Test that the Agentic AI prerequisite diagnostic
    retrieves grounded context from the current
    Python and Machine Learning curricula.
    """

    retrieved_topic_ids = []
    retrieval_queries = []

    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        retrieved_topic_ids.append(
            topic_id
        )

        retrieval_queries.append(
            retrieval_query
        )

        return (
            f"Learning material for {topic_name}."
        )

    class MockGenerateQuiz:
        def invoke(self, inputs):
            topic_id = inputs["topics"][0][
                "topic_id"
            ]

            return json.dumps(
                {
                    "questions": [
                        {
                            "topic_id": topic_id,
                            "question_text": (
                                f"Question {i}"
                            ),
                            "question_type": (
                                "multiple_choice"
                            ),
                            "difficulty": "beginner",
                            "options": [
                                "A",
                                "B",
                                "C",
                                "D",
                            ],
                            "correct_answer": "A",
                        }
                        for i in range(
                            1,
                            inputs["num_questions"] + 1
                        )
                    ]
                }
            )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    monkeypatch.setattr(
        "app.agent.graph.generate_quiz",
        MockGenerateQuiz()
    )

    state = {
        "selected_path": "agentic_ai",
        "learner_context": {
            "current_level": "beginner"
        },
    }

    with SessionLocal() as db:
        result = generate_initial_diagnostic_node(
            state,
            db
        )

    with SessionLocal() as db:
        expected_python_topic_ids = [
            get_topic_id_by_key(
                db,
                f"python_ch{chapter_number}_1",
            )
            for chapter_number in range(1, 9)
        ]

        expected_ml_topic_ids = [
            get_topic_id_by_key(
                db,
                f"machine_learning_ch{chapter_number}_1",
            )
            for chapter_number in range(1, 8)
        ]

    assert retrieved_topic_ids == (
        expected_python_topic_ids
        + expected_ml_topic_ids
    )

    python_query = (
        "Python fundamentals including variables, data types, "
        "operators, strings, conditions, loops, collections, "
        "functions, and basic Python programming concepts"
    )

    ml_query = (
        "Machine learning fundamentals including supervised "
        "learning, unsupervised learning, deep learning, "
        "reinforcement learning, ensemble learning, "
        "classification, regression, clustering, model training, "
        "and evaluation"
    )

    assert retrieval_queries == (
        [python_query] * 8
        + [ml_query] * 7
    )

    assert len(
        result["assessment_questions"]
    ) == 10


def test_submit_initial_diagnostic_node(monkeypatch):
    """
    Test that submitting the initial diagnostic:
    1. Evaluates the learner's answers.
    2. Saves the diagnostic assessment.
    3. Creates the personalized learning path.
    4. Selects the learner's first topic.
    """

    class FakeEvaluationTool:
        def invoke(self, inputs):
            return json.dumps(
                {
                    "is_correct": True,
                    "score_awarded": 1,
                    "feedback": "Correct.",
                }
            )

    class FakeAttempt:
        assessment_attempt_id = 10
        score = 2
        max_score = 2

    # Prevent the test from calling the real LLM.
    monkeypatch.setattr(
        "app.agent.graph.evaluate_answer",
        FakeEvaluationTool()
    )

    # Mock saving the diagnostic assessment.
    def fake_save_assessment_result(
        db,
        user_id,
        topic_id,
        assessment_type,
        questions,
        feedback=None,
    ):
        assert user_id == 2
        assert topic_id is None
        assert assessment_type == "diagnostic"
        assert len(questions) == 2

        # Make sure the topic IDs are preserved
        # for topic-level mastery calculation.
        assert questions[0]["topic_id"] == 1
        assert questions[1]["topic_id"] == 2

        assert questions[0]["is_correct"] is True
        assert questions[1]["is_correct"] is True

        return FakeAttempt()

    monkeypatch.setattr(
        "app.agent.graph.save_assessment_result",
        fake_save_assessment_result
    )

    # Mock personalized learning path creation.
    def fake_create_learning_path(
        db,
        user_id,
        target_topic_id,
        path_name,
        goal,
    ):
        assert user_id == 2
        assert target_topic_id == get_topic_id_by_key(
            db,
            "agentic_ai_ch5_3",
        )
        assert path_name == "Agentic AI Learning Path"
        assert goal == "Learn Agentic AI"

        return {
            "learning_path_id": 5,
            "user_id": 2,
            "name": "Agentic AI Learning Path",
            "goal": "Learn Agentic AI",
            "status": "active",
            "topic_ids": list(range(1, 26)),
        }

    monkeypatch.setattr(
        "app.agent.graph.create_learning_path",
        fake_create_learning_path
    )

    # Mock selection of the first topic
    # that the learner still needs to study.
    def fake_select_next_topic(
        db,
        learning_path_id,
    ):
        assert learning_path_id == 5

        return {
            "topic_id": 3,
            "name": "Introduction to Agentic AI",
            "status": "pending",
            "recommended_action": "explain",
        }

    monkeypatch.setattr(
        "app.agent.graph.select_next_topic",
        fake_select_next_topic
    )

    state = {
        "user_id": 2,
        "selected_path": "agentic_ai",
        "assessment_questions": [
            {
                "topic_id": 1,
                "question_text": "What is a Python variable?",
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
            },
            {
                "topic_id": 2,
                "question_text": "What is supervised learning?",
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "B",
            },
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            },
            {
                "learner_answer": "B",
            },
        ],
    }

    with SessionLocal() as db:
        result = submit_initial_diagnostic_node(
            state,
            db
        )

    assert result["assessment_result"] == {
        "assessment_attempt_id": 10,
        "score": 2,
        "max_score": 2,
    }

    assert result["learning_path"]["learning_path_id"] == 5

    assert result["current_topic"]["topic_id"] == 3
    assert result["next_topic_id"] == 3

    assert (
        result["response"]
        == "Your diagnostic assessment has been completed "
           "and your personalized learning path is ready."
    )


def test_submit_initial_diagnostic_without_selected_path():
    """
    Test that the diagnostic cannot be submitted
    without a selected learning path.
    """

    with SessionLocal() as db:
        state = {
            "user_id": 2,
            "assessment_questions": [],
            "assessment_answers": [],
        }

        result = submit_initial_diagnostic_node(
            state,
            db
        )

    assert "select a learning path" in result["response"].lower()


def test_submit_initial_diagnostic_unconfigured_path(
    monkeypatch
):
    """
    Test that a learning path without a target topic
    cannot create a personalized learning path.
    """

    monkeypatch.setitem(
        AVAILABLE_LEARNING_PATHS,
        "machine_learning",
        {
            "name": "Machine Learning",
            "diagnostic_topic_keys": ["agentic_ai_ch1_1"],
            "target_topic_key": None,
        },
    )

    with SessionLocal() as db:
        state = {
            "user_id": 2,
            "selected_path": "machine_learning",
        }

        result = submit_initial_diagnostic_node(
            state,
            db
        )

    assert (
        result["response"]
        == "This learning path is not fully configured yet."
    )


def test_submit_initial_diagnostic_without_answers():
    """
    Test that diagnostic submission requires
    both questions and learner answers.
    """

    with SessionLocal() as db:
        state = {
            "user_id": 2,
            "selected_path": "agentic_ai",
            "assessment_questions": [
                {
                    "topic_id": 1,
                    "question_text": "What is a Python variable?",
                    "correct_answer": "A",
                }
            ],
            "assessment_answers": [],
        }

        result = submit_initial_diagnostic_node(
            state,
            db
        )

    assert (
        "questions and answers are required"
        in result["response"].lower()
    )


def test_submit_initial_diagnostic_incomplete_answers():
    """
    Test that every diagnostic question
    must have a learner answer.
    """

    with SessionLocal() as db:
        state = {
            "user_id": 2,
            "selected_path": "agentic_ai",
            "assessment_questions": [
                {
                    "topic_id": 1,
                    "question_text": "Question 1",
                    "correct_answer": "A",
                },
                {
                    "topic_id": 2,
                    "question_text": "Question 2",
                    "correct_answer": "B",
                },
            ],
            "assessment_answers": [
                {
                    "learner_answer": "A",
                }
            ],
        }

        result = submit_initial_diagnostic_node(
            state,
            db
        )

    assert (
        "answer all diagnostic questions"
        in result["response"].lower()
    )


def test_route_tutor_entry_to_normal_tutor():
    """
    Test that an existing learner without a newly
    selected path continues to the normal tutor flow.
    """

    state = {
        "user_id": 2,
        "learner_context": {
            "current_level": "beginner"
        },
    }

    result = route_tutor_entry(state)

    assert result == "continue_tutor"


def test_route_tutor_entry_to_generate_diagnostic():
    """
    Test that selecting a new learning path
    starts the initial diagnostic generation flow.
    """

    state = {
        "user_id": 2,
        "selected_path": "agentic_ai",
    }

    result = route_tutor_entry(state)

    assert result == "generate_initial_diagnostic"


def test_route_tutor_entry_to_submit_diagnostic():
    """
    Test that diagnostic questions and answers
    route the learner to diagnostic submission.
    """

    state = {
        "user_id": 2,
        "selected_path": "agentic_ai",
        "assessment_questions": [
            {
                "topic_id": 1,
                "question_text": "What is a Python variable?",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "A",
            }
        ],
    }

    result = route_tutor_entry(state)

    assert result == "submit_initial_diagnostic"


def test_tutor_graph_initial_diagnostic_flow(
    monkeypatch
):
    """
    Test that the compiled Tutor Agent graph creates
    the Python learning path directly when no
    prerequisite diagnostic is required.
    """

    monkeypatch.setattr(
        "app.agent.graph.create_learning_path",
        lambda db,
        user_id,
        target_topic_id,
        path_name,
        goal: {
            "learning_path_id": 100,
            "user_id": user_id,
            "name": path_name,
            "goal": goal,
            "status": "active",
            "topic_ids": list(range(35, 43)),
        }
    )

    monkeypatch.setattr(
        "app.agent.graph.select_next_topic",
        lambda db, learning_path_id: {
            "topic_id": 35,
            "name": "Ch1 Introduction To Python",
            "status": "pending",
            "recommended_action": "explain",
        }
    )

    graph = build_tutor_graph()

    state = {
        "user_id": 2,
        "selected_path": "python",
    }

    result = graph.invoke(
        state
    )

    assert result["selected_path"] == "python"
    assert result["assessment_type"] == "diagnostic"
    assert result["diagnostic_topics"] == []
    assert result["assessment_questions"] == []

    assert (
        result["learning_path"]["learning_path_id"]
        == 100
    )

    assert result["next_topic_id"] == 35

    assert (
        result["response"]
        == (
            "No prerequisite diagnostic assessment "
            "is required. Your learning path is ready."
        )
    )

def test_route_tutor_entry_to_assessment_submission():
    """
    Topic assessment questions and answers should
    route directly to assessment submission.
    """

    state = {
        "assessment_questions": [
            {
                "topic_id": 27,
                "question_text": (
                    "What value is assigned to a?"
                ),
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "4",
            }
        ],
    }

    result = route_tutor_entry(state)

    assert result == "submit_assessment"


def test_route_tutor_entry_continues_normal_tutor():
    """
    A normal learner message should continue
    through the Tutor conversation workflow.
    """

    state = {
        "user_message": (
            "Explain Python variables to me."
        ),
    }

    result = route_tutor_entry(state)

    assert result == "continue_tutor"


def test_prepare_personalized_inputs_with_personalization():
    """
    Teaching inputs should include learner mastery,
    preferences, conversation history, and current request.
    """

    state = {
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "practical_examples",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 20.0,
            "weak_areas": [
                {
                    "area": "Variable reassignment",
                    "reason": "Incorrect understanding.",
                },
                {
                    "area": "Multiple assignment order",
                    "reason": "Values were reversed.",
                },
            ],
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "I find variables confusing.",
            }
        ],
        "user_message": "Explain Python variables.",
    }

    result = prepare_personalized_inputs(state)

    assert result["topic_id"] == 27

    assert (
        result["topic_name"]
        == "Python Reference Material"
    )

    assert (
        result["student_level"]
        == "beginner"
    )

    assert (
        result["mastery_score"]
        == 20.0
    )

    assert len(
        result["weak_areas"]
    ) == 2

    assert (
        result["weak_areas"][0]["area"]
        == "Variable reassignment"
    )

    assert (
        result["preferred_format"]
        == "practical_examples"
    )

    assert (
        result["preferred_pace"]
        == "slow"
    )

    assert len(
        result["conversation_history"]
    ) == 1

    assert (
        result["user_message"]
        == "Explain Python variables."
    )


def test_practice_node_uses_learner_preferences(
    monkeypatch
):
    """
    The practice node should include learner
    preferences in the personalized practice request.
    """

    # Mock RAG retrieval so the test does not
    # call the real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "Python variables store and update values."
        )
    )

    captured_inputs = {}

    class FakePracticeTool:
        def invoke(self, inputs):
            # Capture the inputs sent to the practice tool.
            captured_inputs.update(inputs)

            return (
                "Personalized practice generated."
            )

    monkeypatch.setattr(
        "app.agent.graph.generate_practice",
        FakePracticeTool()
    )

    state = {
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "practical_examples",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 55,
            "weak_areas": [
                "Variable reassignment",
            ],
        },
        "user_message": (
            "Give me practice about variables."
        ),
    }

    result = practice_node(state)

    personalized_topic = captured_inputs[
        "topic"
    ]

    assert (
        "55%"
        in personalized_topic
    )

    assert (
        "practical_examples"
        in personalized_topic
    )

    assert (
        "slow"
        in personalized_topic
    )

    assert (
        "Give me practice about variables."
        in personalized_topic
    )

    assert json.loads(
        captured_inputs["weak_areas"]
    ) == [
        "Variable reassignment",
    ]

    assert (
        result["response"]
        == "Personalized practice generated."
    )



def test_select_practice_type_python_low_mastery():
    """
    Low-mastery Python learners should receive
    simple concept-reinforcement practice.
    """

    result = select_practice_type(
        topic_name="Python Foundations",
        mastery_score=30,
        preferred_format=None,
    )

    assert result == "flashcards"

def test_select_practice_type_python_medium_mastery():
    """
    Medium-mastery Python learners should receive
    short-answer practice.
    """

    result = select_practice_type(
        topic_name="Python Functions",
        mastery_score=60,
        preferred_format=None,
    )

    assert result == "short_answer"


def test_select_practice_type_python_high_mastery():
    """
    High-mastery Python learners should receive
    application-oriented coding practice.
    """

    result = select_practice_type(
        topic_name="Python Functions",
        mastery_score=90,
        preferred_format=None,
    )

    assert result == "coding"


def test_select_practice_type_agentic_medium_mastery():
    """
    Medium-mastery Agentic AI learners should receive
    conceptual short-answer practice.
    """

    result = select_practice_type(
        topic_name="Building Your First Agent",
        mastery_score=60,
        preferred_format=None,
    )

    assert result == "short_answer"


def test_select_practice_type_agentic_high_mastery():
    """
    Higher-mastery Agentic AI learners should receive
    applied scenario-based practice.
    """

    result = select_practice_type(
        topic_name="Agent Tool Use",
        mastery_score=80,
        preferred_format=None,
    )

    assert result == "scenario"


def test_select_practice_type_guided_python_preference():
    """
    Guided-practice preference should favor
    coding practice for programming topics.
    """

    result = select_practice_type(
        topic_name="Python Variables",
        mastery_score=75,
        preferred_format="guided_practice",
    )

    assert result == "coding"


def test_select_practice_type_practical_agentic_preference():
    """
    Practical-example preference should favor
    scenarios for Agentic AI topics.
    """

    result = select_practice_type(
        topic_name="Agent Planning",
        mastery_score=55,
        preferred_format="practical_examples",
    )

    assert result == "scenario"


def test_select_practice_type_without_mastery():
    """
    Learners without mastery data should receive
    a safe default practice type.
    """

    result = select_practice_type(
        topic_name="Introduction to LLMs",
        mastery_score=None,
        preferred_format=None,
    )

    assert result == "flashcards"


def test_review_node_uses_personalization_data(
    monkeypatch
):
    """
    The review node should include learner mastery,
    structured weak areas, preferences, and user request
    in the personalized review prompt.
    """

    # Mock RAG retrieval so the test does not
    # call the real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "Python variables can be reassigned "
            "during program execution."
        )
    )

    captured_inputs = {}

    class FakeExplainTool:
        def invoke(self, inputs):
            # Capture the inputs sent to the explanation tool.
            captured_inputs.update(inputs)

            return (
                "Personalized review generated."
            )

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "practical_examples",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                {
                    "area": "Variable reassignment",
                    "reason": "Incorrect understanding.",
                },
                {
                    "area": "Multiple assignment order",
                    "reason": "Values were reversed.",
                },
            ],
        },
        "user_message": (
            "Review Python variables with me."
        ),
    }

    result = review_node(state)

    review_topic = captured_inputs[
        "topic"
    ]

    assert (
        "75%"
        in review_topic
    )

    assert (
        "Variable reassignment"
        in review_topic
    )

    assert (
        "Multiple assignment order"
        in review_topic
    )

    assert (
        "practical_examples"
        in review_topic
    )

    assert (
        "slow"
        in review_topic
    )

    assert (
        "Review Python variables with me."
        in review_topic
    )

    assert (
        captured_inputs["student_level"]
        == "beginner"
    )

    assert (
        result["response"]
        == "Personalized review generated."
    )


def test_generate_assessment_node_uses_personalization(
    monkeypatch
):
    """
    The assessment node should include learner mastery,
    weak areas, preferences, and request in the
    personalized assessment topic.
    """

    # Mock RAG retrieval so the test does not
    # call the real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "Python variables store values and can "
            "be reassigned during program execution."
        )
    )

    captured_inputs = {}

    class FakeQuizTool:
        def invoke(self, inputs):
            # Capture inputs sent to the quiz tool.
            captured_inputs.update(inputs)

            fake_quiz = {
                "questions": [
                    {
                        "topic_id": 27,
                        "question_text": (
                            "What happens when a variable "
                            "is reassigned?"
                        ),
                        "question_type": "multiple_choice",
                        "difficulty": "beginner",
                        "options": [
                            "A",
                            "B",
                            "C",
                            "D",
                        ],
                        "correct_answer": "A",
                    }
                ]
            }

            return json.dumps(
                fake_quiz
            )

    monkeypatch.setattr(
        "app.agent.graph.generate_quiz",
        FakeQuizTool()
    )

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 27,
            "name": "Python Reference Material",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "practical_examples",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                {
                    "area": "Variable reassignment",
                    "reason": "Incorrect understanding.",
                },
                {
                    "area": "Multiple assignment order",
                    "reason": "Values were reversed.",
                },
            ],
        },
        "assessment_type": "topic",
        "user_message": (
            "Assess my understanding of variables."
        ),
    }

    result = generate_assessment_node(
        state
    )

    personalized_topic = (
        captured_inputs["topics"][0]["topic"]
    )

    assert (
        'Assess the topic "Python Reference Material".'
        in personalized_topic
    )

    assert (
        "75%"
        in personalized_topic
    )

    assert (
        "Variable reassignment"
        in personalized_topic
    )

    assert (
        "Multiple assignment order"
        in personalized_topic
    )

    assert (
        "practical_examples"
        in personalized_topic
    )

    assert (
        "slow"
        in personalized_topic
    )

    assert (
        "Assess my understanding of variables."
        in personalized_topic
    )

    assert (
        captured_inputs["student_level"]
        == "beginner"
    )

    assert (
        captured_inputs["assessment_type"]
        == "topic"
    )

    assert (
        result["assessment_questions"][0][
            "topic_id"
        ]
        == 27
    )

def test_submit_assessment_node_generates_personalized_feedback(
    monkeypatch
):
    """
    Assessment submission should generate personalized
    feedback using assessment results, weak areas,
    and learner level.
    """

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
        },
        "assessment_type": "topic",
        "assessment_questions": [
            {
                "topic_id": 12,
                "question_text": "What is an AI agent?",
                "question_type": "multiple_choice",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
            }
        ],
        "assessment_answers": [
            {
                "learner_answer": "B",
            }
        ],
    }

    evaluated_responses = [
        {
            "topic_id": 12,
            "question_text": "What is an AI agent?",
            "question_type": "multiple_choice",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "learner_answer": "B",
            "is_correct": False,
            "score_awarded": 0,
            "feedback": "Review the definition of an AI agent.",
        }
    ]

    monkeypatch.setattr(
        "app.agent.graph.evaluate_assessment_responses",
        lambda responses: evaluated_responses
    )

    class FakeAssessmentAttempt:
        assessment_attempt_id = 100
        score = 0
        max_score = 1
        feedback = None
    fake_attempt = FakeAssessmentAttempt()

    monkeypatch.setattr(
        "app.agent.graph.save_assessment_result",
        lambda db, user_id, topic_id,
        assessment_type, questions: fake_attempt
    )

    updated_mastery = {
        "mastery_score": 40.0,
        "weak_areas": [
            {
                "area": "Agent definition",
                "reason": "Incorrect answer.",
            }
        ],
        "last_assessed_at": None,
    }

    monkeypatch.setattr(
        "app.agent.graph.get_topic_mastery",
        lambda db, user_id, topic_id: updated_mastery
    )

    monkeypatch.setattr(
        "app.agent.graph.get_active_learning_path",
        lambda db, user_id: None
    )

    captured_feedback_inputs = {}

    class FakeFeedbackTool:
        def invoke(self, inputs):
            captured_feedback_inputs.update(
                inputs
            )

            return json.dumps(
                {
                    "summary": (
                        "The learner needs more review."
                    ),
                    "strengths": [],
                    "weak_areas": [
                        "Agent definition"
                    ],
                    "recommendation": (
                        "Review the core concept."
                    ),
                }
            )

    monkeypatch.setattr(
        "app.agent.graph.generate_feedback",
        FakeFeedbackTool()
    )

    result = submit_assessment_node(
        state
    )

    assert (
        captured_feedback_inputs["student_level"]
        == "intermediate"
    )

    assert (
        "Agent definition"
        in captured_feedback_inputs["weak_areas"]
    )

    assert (
        "What is an AI agent?"
        in captured_feedback_inputs[
            "assessment_results"
        ]
    )

    assert (
        result["assessment_feedback"]["summary"]
        == "The learner needs more review."
    )

    assert (
        result["assessment_feedback"]["recommendation"]
        == "Review the core concept."
    )
    assert (
        "The learner needs more review."
        in result["response"]
    )

    assert (
        "Areas to review:"
        in result["response"]
    )

    assert (
        "Agent definition"
        in result["response"]
    )

    assert (
        "Review the core concept."
        in result["response"]
    )

    assert (
        fake_attempt.feedback
        is not None
    )

    assert (
        "Agent definition"
        in fake_attempt.feedback
    )


def test_teach_node_uses_conversation_history(
    monkeypatch
):
    """
    The teaching node should include recent
    conversation history in the personalized
    explanation request.
    """

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "An AI agent can observe, reason, "
            "and take actions toward a goal."
        )
    )

    captured_inputs = {}

    class FakeExplainTool:
        def invoke(self, inputs):
            captured_inputs.update(inputs)

            return "Personalized explanation."

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "detailed_explanations",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 40,
            "weak_areas": [],
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "What is an AI agent?",
            },
            {
                "role": "assistant",
                "content": (
                    "An AI agent can make decisions "
                    "and take actions."
                ),
            },
        ],
        "user_message": (
            "Can you explain that in more detail?"
        ),
    }

    result = teach_node(
        state
    )

    personalized_topic = captured_inputs[
        "topic"
    ]

    assert (
        "Recent conversation context:"
        in personalized_topic
    )

    assert (
        "user: What is an AI agent?"
        in personalized_topic
    )

    assert (
        "assistant: An AI agent can make decisions"
        in personalized_topic
    )

    assert (
        "Can you explain that in more detail?"
        in personalized_topic
    )

    assert (
        result["response"]
        == "Personalized explanation."
    )

def test_practice_node_uses_conversation_history(
    monkeypatch
):
    """
    The practice node should include recent
    conversation history in the personalized
    practice request.
    """

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "Python functions can accept parameters "
            "and return values."
        )
    )

    captured_inputs = {}

    class FakePracticeTool:
        def invoke(self, inputs):
            captured_inputs.update(inputs)

            return json.dumps(
                {
                    "practice_type": inputs["practice_type"],
                    "items": [
                        {
                            "prompt": "Complete the function.",
                            "answer": "return x",
                        }
                        for _ in range(5)
                    ],
                }
            )

    monkeypatch.setattr(
        "app.agent.graph.generate_practice",
        FakePracticeTool()
    )

    state = {
        "current_topic": {
            "topic_id": 27,
            "name": "Python Foundations",
        },
        "learner_context": {
            "current_level": "beginner",
            "preferred_format": "guided_practice",
            "preferred_pace": "slow",
        },
        "topic_mastery": {
            "mastery_score": 60,
            "weak_areas": [],
        },
        "conversation_history": [
            {
                "role": "user",
                "content": "Explain Python functions.",
            },
            {
                "role": "assistant",
                "content": (
                    "A function is a reusable block of code."
                ),
            },
        ],
        "user_message": (
            "Give me practice on what you just explained."
        ),
    }

    result = practice_node(
        state
    )

    personalized_topic = captured_inputs[
        "topic"
    ]

    assert (
        "Recent conversation context:"
        in personalized_topic
    )

    assert (
        "user: Explain Python functions."
        in personalized_topic
    )

    assert (
        "assistant: A function is a reusable block of code."
        in personalized_topic
    )

    assert (
        "Give me practice on what you just explained."
        in personalized_topic
    )

    assert (
        result["response"]
        is not None
    )


def test_review_node_uses_conversation_history(
    monkeypatch
):
    """
    The review node should include recent
    conversation history in the personalized
    review request.
    """

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None: (
            "An AI agent can observe, reason, "
            "and act toward a goal."
        )
    )

    captured_inputs = {}

    class FakeExplainTool:
        def invoke(self, inputs):
            captured_inputs.update(inputs)

            return "Personalized review."

    monkeypatch.setattr(
        "app.agent.graph.explain",
        FakeExplainTool()
    )

    state = {
        "current_topic": {
            "topic_id": 12,
            "name": "Building Your First Agent",
        },
        "learner_context": {
            "current_level": "intermediate",
            "preferred_format": "detailed_explanations",
            "preferred_pace": "moderate",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                {
                    "area": "Agent reasoning",
                    "reason": "Needs reinforcement.",
                }
            ],
        },
        "conversation_history": [
            {
                "role": "user",
                "content": (
                    "What is the difference between "
                    "reasoning and action?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Reasoning decides what to do, "
                    "while action executes it."
                ),
            },
        ],
        "user_message": (
            "Review what we discussed earlier."
        ),
    }

    result = review_node(
        state
    )

    review_topic = captured_inputs[
        "topic"
    ]

    assert (
        "Recent conversation context:"
        in review_topic
    )

    assert (
        "user: What is the difference between "
        "reasoning and action?"
        in review_topic
    )

    assert (
        "assistant: Reasoning decides what to do"
        in review_topic
    )

    assert (
        "Review what we discussed earlier."
        in review_topic
    )

    assert (
        result["response"]
        == "Personalized review."
    )

@pytest.mark.parametrize(
    (
        "preferred_format",
        "topic_name",
        "mastery_score",
        "expected_type",
    ),
    [
        (
            "Concise Explanations",
            "Python Variables",
            90,
            "flashcards",
        ),
        (
            "Detailed Explanations",
            "Python Variables",
            20,
            "short_answer",
        ),
        (
            "Practical Examples",
            "Python Variables",
            20,
            "coding",
        ),
        (
            "Guided Practice",
            "Agent Planning",
            20,
            "scenario",
        ),
        (
            "Explanations with Examples",
            "Python Variables",
            90,
            "short_answer",
        ),
    ],
)
def test_select_practice_type_accepts_frontend_labels(
    preferred_format,
    topic_name,
    mastery_score,
    expected_type,
):
    """
    Frontend labels and the legacy label should map
    to the practice types used internally by the agent.
    """

    result = select_practice_type(
        topic_name=topic_name,
        mastery_score=mastery_score,
        preferred_format=preferred_format,
    )

    assert result == expected_type


def test_practice_uses_existing_topic_explanation(
    monkeypatch
):
    """
    Test that practice generation uses the existing
    topic explanation without calling RAG again.
    """

    # Fail the test if retrieval is called.
    def fail_if_retrieval_called(
        topic_id,
        topic_name,
        retrieval_query=None,
    ):
        raise AssertionError(
            "RAG retrieval should not be called "
            "when topic_explanation is available."
        )

    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        fail_if_retrieval_called,
    )

    class FakePracticeTool:
        def invoke(self, inputs):
            return {
                "practice_type": "flashcards",
                "items": [
                    {
                        "prompt": "What is Python?",
                        "answer": "A programming language.",
                    }
                ],
            }

    monkeypatch.setattr(
        "app.agent.graph.generate_practice",
        FakePracticeTool(),
    )

    state = {
        "user_id": 2,
        "user_message": "Give me practice.",
        "topic_explanation": (
            "Python is a high-level programming language."
        ),
        "current_topic": {
            "topic_id": 35,
            "name": "Introduction to Python",
        },
        "learner_context": {
            "current_level": "Beginner",
        },
        "topic_mastery": {},
    }

    result = practice_node(state)

    assert result["last_action"] == "practice"
    assert result["response"]

def test_extract_practice_item_count_from_word():
    """
    A number written as a word should determine
    the requested number of practice items.
    """

    result = extract_practice_item_count(
        "Give me one short practice question about this topic."
    )

    assert result == 1


def test_extract_practice_item_count_from_digit():
    """
    A numeric quantity should determine
    the requested number of practice items.
    """

    result = extract_practice_item_count(
        "Give me 3 practice questions."
    )

    assert result == 3


def test_extract_practice_item_count_uses_default():
    """
    Practice requests without a quantity
    should use the default number of items.
    """

    result = extract_practice_item_count(
        "Give me some practice about this topic."
    )

    assert result == 5


def test_extract_practice_item_count_limits_large_request():
    """
    Large practice requests should be limited
    to prevent excessive generation.
    """

    result = extract_practice_item_count(
        "Give me 50 practice questions."
    )

    assert result == 10


def test_select_practice_type_explicit_scenario_overrides_preference():
    """
    Test that an explicit scenario request has priority
    over the learner's stored preferred format.
    """

    result = select_practice_type(
        topic_name="Prompt Engineering",
        mastery_score=30,
        preferred_format="concise_explanations",
        user_message="Give me 2 scenarios about prompt optimization",
    )

    assert result == "scenario"


def test_select_practice_type_explicit_flashcards_overrides_preference():
    """
    Test that an explicit flashcard request has priority
    over the learner's stored preferred format.
    """

    result = select_practice_type(
        topic_name="Prompt Engineering",
        mastery_score=90,
        preferred_format="detailed_explanations",
        user_message="Give me 3 flashcards about prompt engineering",
    )

    assert result == "flashcards"


def test_select_practice_type_explicit_coding_overrides_preference():
    """
    Test that an explicit coding request has priority
    over the learner's stored preferred format.
    """

    result = select_practice_type(
        topic_name="Python Functions",
        mastery_score=30,
        preferred_format="concise_explanations",
        user_message="Give me coding practice about functions",
    )

    assert result == "coding"


def test_route_topic_scope_allows_generic_current_topic_request(
    monkeypatch,
):
    """
    Generic practice requests should stay within
    the learner's current topic.
    """

    monkeypatch.setattr(
        "app.agent.graph.classify_topic_scope",
        lambda user_message, current_topic_name: "no_explicit_topic",
    )

    state = {
        "user_message": "Give me 3 flashcards",
        "current_topic": {
            "topic_id": 6,
            "name": (
                "Agentic AI | Ch1.3 – Prompt engineering "
                "and LLM prompt optimization"
            ),
        },
    }

    result = route_topic_scope(state)

    assert result == "continue"


def test_route_topic_scope_allows_current_topic_subtopic(
    monkeypatch,
):
    """
    Requests about a subtopic of the current topic
    should continue through the Tutor workflow.
    """

    monkeypatch.setattr(
        "app.agent.graph.classify_topic_scope",
        lambda user_message, current_topic_name: "in_scope",
    )

    state = {
        "user_message": "Explain zero-shot prompting more",
        "current_topic": {
            "topic_id": 6,
            "name": (
                "Agentic AI | Ch1.3 – Prompt engineering "
                "and LLM prompt optimization"
            ),
        },
    }

    result = route_topic_scope(state)

    assert result == "continue"


def test_route_topic_scope_blocks_different_topic_request(
    monkeypatch,
):
    """
    Explicit requests about another topic should
    be blocked by the topic-scope guard.
    """

    monkeypatch.setattr(
        "app.agent.graph.classify_topic_scope",
        lambda user_message, current_topic_name: "out_of_scope",
    )

    state = {
        "user_message": "Give me 3 flashcards about RAG",
        "current_topic": {
            "topic_id": 6,
            "name": (
                "Agentic AI | Ch1.3 – Prompt engineering "
                "and LLM prompt optimization"
            ),
        },
    }

    result = route_topic_scope(state)

    assert result == "out_of_scope"


def test_route_topic_scope_blocks_assessment_on_different_topic(
    monkeypatch,
):
    """
    Assessment requests about another topic should
    also be blocked by the topic-scope guard.
    """

    monkeypatch.setattr(
        "app.agent.graph.classify_topic_scope",
        lambda user_message, current_topic_name: "out_of_scope",
    )

    state = {
        "user_message": "Quiz me on Python",
        "current_topic": {
            "topic_id": 6,
            "name": (
                "Agentic AI | Ch1.3 – Prompt engineering "
                "and LLM prompt optimization"
            ),
        },
    }

    result = route_topic_scope(state)

    assert result == "out_of_scope"


def test_out_of_scope_node_returns_guard_message():
    """
    The out-of-scope node should tell the learner
    to stay within the current learning topic.
    """

    state = {
        "user_message": "Give me 3 flashcards about RAG",
        "current_topic": {
            "topic_id": 6,
            "name": (
                "Agentic AI | Ch1.3 – Prompt engineering "
                "and LLM prompt optimization"
            ),
        },
    }

    result = out_of_scope_node(state)

    assert "response" in result

    assert (
        "outside your current learning topic"
        in result["response"]
    )

    assert (
        "Prompt engineering"
        in result["response"]
    )

    # The guard message is not a learning action,
    # so it should not overwrite the last Tutor action.
    assert "last_action" not in result


def test_tutor_graph_stops_out_of_scope_request(
    monkeypatch,
):
    """
    An out-of-scope learner request should stop
    before any Tutor content action is executed.
    """

    # Force the scope classifier to reject the request.
    monkeypatch.setattr(
        "app.agent.graph.classify_topic_scope",
        lambda user_message, current_topic_name: "out_of_scope",
    )

    # Fail the test immediately if practice is reached.
    def fail_if_practice_runs(state):
        raise AssertionError(
            "Practice node should not run "
            "for an out-of-scope request."
        )

    monkeypatch.setattr(
        "app.agent.graph.practice_node",
        fail_if_practice_runs,
    )

    from app.agent.graph import build_tutor_graph

    graph = build_tutor_graph()

    result = graph.invoke(
        {
            "user_id": 2,
            "session_id": 409,
            "user_message": (
                "Give me 3 flashcards about RAG"
            ),
        }
    )

    assert (
        "outside your current learning topic"
        in result["response"]
    )

def test_tutor_graph_blocks_out_of_scope_request(
    monkeypatch,
):
    """
    The complete Tutor Agent workflow should stop
    at the topic-scope guard when the learner asks
    about a different topic.
    """

    # Force the topic-scope classifier to reject
    # the learner's request deterministically.
    monkeypatch.setattr(
        "app.agent.graph.classify_topic_scope",
        lambda user_message, current_topic_name: "out_of_scope",
    )

    # Practice must never run for an out-of-scope request.
    def fail_if_practice_runs(state):
        raise AssertionError(
            "Practice node should not run "
            "for an out-of-scope request."
        )

    monkeypatch.setattr(
        "app.agent.graph.practice_node",
        fail_if_practice_runs,
    )

    with SessionLocal() as db:
        session = None

        try:
            # Load the learner's active learning path so
            # the temporary session belongs to a real path.
            learning_path = get_active_learning_path(
                db=db,
                user_id=2,
            )

            assert learning_path is not None

            learning_path_id = learning_path[
                "learning_path_id"
            ]

            # Select the learner's current available topic.
            current_topic = select_next_topic(
                db=db,
                learning_path_id=learning_path_id,
            )

            assert current_topic is not None

            # Create a temporary chat session for
            # the current learning-path topic.
            session = ChatSession(
                user_id=2,
                learning_path_id=learning_path_id,
                topic_id=current_topic["topic_id"],
                session_name="Topic Scope Guard Test",
                started_at=datetime.now(timezone.utc),
            )

            db.add(session)
            db.commit()
            db.refresh(session)

            # Build the complete Tutor Agent workflow.
            graph = build_tutor_graph()

            result = graph.invoke(
                {
                    "user_id": 2,
                    "session_id": session.chat_session_id,
                    "user_message": (
                        "Give me 3 flashcards about RAG"
                    ),
                }
            )

            # The request should stop at the scope guard.
            assert (
                "outside your current learning topic"
                in result["response"]
            )

            # No learning action should be recorded
            # because no actual Tutor action was executed.
            assert result.get("last_action") is None

            # The learner should remain on the same topic.
            assert (
                result["current_topic"]["topic_id"]
                == current_topic["topic_id"]
            )

        finally:
            # Remove the temporary chat session
            # without changing learner progress.
            if session is not None:
                db.query(ChatMessage).filter(
                    ChatMessage.session_id
                    == session.chat_session_id
                ).delete()

                db.delete(session)
                db.commit()


def test_plan_next_topic_node_keeps_existing_session_topic():
    """
    A topic already attached to the chat session
    should not be replaced by normal path planning.
    """

    state = {
        "user_id": 2,
        "current_topic": {
            "topic_id": 6,
            "name": (
                "Agentic AI | Ch1.3 – Prompt engineering "
                "and LLM prompt optimization"
            ),
        },
    }

    result = plan_next_topic_node(state)

    assert result == {}


def test_plan_next_topic_node_plans_when_no_session_topic(
    monkeypatch,
):
    """
    Normal path planning should still run when
    no topic is already attached to the session.
    """

    expected_topic = {
        "topic_id": 6,
        "name": (
            "Agentic AI | Ch1.3 – Prompt engineering "
            "and LLM prompt optimization"
        ),
    }

    monkeypatch.setattr(
        "app.agent.graph.plan_next_topic",
        lambda state, db: {
            "current_topic": expected_topic
        },
    )

    state = {
        "user_id": 2,
        "learning_path": {
            "learning_path_id": 123,
        },
    }

    result = plan_next_topic_node(state)

    assert result["current_topic"] == expected_topic
