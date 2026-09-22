from datetime import datetime, timezone
import json
from app.database.connection import SessionLocal
from app.agent.graph import (
    load_learner_context,
    load_active_learning_path,
    load_topic_mastery,
    build_tutor_graph,
    update_learning_path_node,
    recommend_node,
    load_conversation_history,
    prepare_assessment_inputs,
    generate_assessment_node,
    prepare_assessment_responses,
    evaluate_assessment_responses,
    submit_assessment_node,
    route_assessment,
    prepare_teaching_inputs,
    teach_node,
    prepare_practice_inputs,
    practice_node,
    prepare_review_inputs,
    review_node,
    generate_initial_diagnostic_node,
    submit_initial_diagnostic_node,
    route_initial_setup,
)
from app.agent.planning import (
    determine_learner_need,
    route_action,
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

from app.agent.recommendation import build_recommendation

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


def test_route_action_explain():
    """
    Test that explain requests are routed
    to the teaching node.
    """

    # Create a state with an explain decision
    state = {
        "learner_need": "explain"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "teach"


def test_route_action_assess():
    """
    Test that assessment requests are routed
    to the assessment node.
    """

    # Create a state with an assess decision
    state = {
        "learner_need": "assess"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "assess"


def test_route_action_practice():
    """
    Test that practice requests are routed
    to the practice node.
    """

    # Create a state with a practice decision
    state = {
        "learner_need": "practice"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "practice"


def test_route_action_review():
    """
    Test that review requests are routed
    to the review node.
    """

    # Create a state with a review decision
    state = {
        "learner_need": "review"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "review"


def test_route_action_recommend():
    """
    Test that recommendation requests are routed
    to the recommendation node.
    """

    # Create a state with a recommend decision
    state = {
        "learner_need": "recommend"
    }

    # Route the action
    result = route_action(state)

    # Verify the correct destination
    assert result == "recommend"


def test_route_action_default():
    """
    Test that an unknown learner need falls back
    to the recommendation node.
    """

    # Create a state with an unexpected value
    state = {
        "learner_need": "unknown"
    }

    # Route the action
    result = route_action(state)

    # Verify the safe default destination
    assert result == "recommend"


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
    Test that the mastery-based recommendation
    is used when the learner asks what to do next.
    """

    # Simulate a learner asking for a recommendation
    state = {
        "learner_need": "recommend",
        "recommended_action": "practice",
    }

    # Resolve the final agent action
    result = resolve_action(state)

    # The mastery-based recommendation should be used
    assert result["recommended_action"] == "practice"


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
    using real learner and learning path data.
    """

    # Mock topic context retrieval so the graph test
    # does not call the real embedding API.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
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

    # Build the compiled LangGraph workflow.
    graph = build_tutor_graph()

    # Create the initial TutorState.
    # The graph should load the remaining information
    # and make the learning recommendation automatically.
    initial_state = {
        "user_id": 2,
        "user_message": "What should I learn next?"
    }

    # Run the complete Tutor Agent workflow.
    result = graph.invoke(initial_state)

    # Verify that conversation history
    # was loaded into TutorState.
    assert "conversation_history" in result

    # No session_id was provided in this test,
    # so the conversation history should be empty.
    assert result["conversation_history"] == []

    # Verify that the learner context
    # was loaded successfully.
    assert result["learner_context"]["user_id"] == 2
    assert result["learner_context"]["goal"] == "Learn Agentic AI"

    # Verify that the active learning path
    # was loaded successfully.
    assert result["learning_path"]["learning_path_id"] == 1
    assert result["learning_path"]["status"] == "active"

    # Verify that the learner request
    # was classified as a recommendation request.
    assert result["learner_need"] == "recommend"

    # Python and Machine Learning are completed,
    # so Agentic AI should be selected next.
    assert result["current_topic"]["topic_id"] == 3
    assert result["current_topic"]["name"] == "Introduction to Agentic AI"

    # Verify that mastery information
    # was loaded for the selected topic.
    assert result["topic_mastery"]["topic_id"] == 3
    assert result["topic_mastery"]["mastery_score"] == 30

    # A mastery score of 30 should result
    # in an explanation recommendation.
    assert result["recommended_action"] == "explain"


def test_tutor_graph_loads_conversation_history(monkeypatch):
    """
    Test that the complete Tutor Agent workflow
    loads recent chat messages into TutorState.
    """
    # Mock topic context retrieval so the graph test
    # does not call the real embedding API.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
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
        lambda topic_id, topic_name: (
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


def test_tutor_graph_routes_high_mastery_to_recommend():
    """
    Test that the Tutor Agent routes a learner
    with high topic mastery to the recommendation node
    and completes the learning path when no topics remain.
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

        # Load the Agentic AI learning path item
        # so its original state can be restored later.
        path_item = (
            db.query(LearningPathItem)
            .filter(
                LearningPathItem.learning_path_id == 1,
                LearningPathItem.topic_id == 3
            )
            .first()
        )

        # Load the learning path so its original
        # status can be restored after the test.
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

        # Store the original database values
        # so the test does not affect other tests.
        original_score = mastery.mastery_score
        original_status = path_item.status
        original_action = path_item.recommended_action
        original_path_status = learning_path.status

        try:
            # Make sure the learning path is active
            # before running the graph.
            learning_path.status = "active"

            # Make sure Agentic AI is the current
            # pending topic before running the graph.
            path_item.status = "pending"
            path_item.recommended_action = "explain"

            # Simulate a successful assessment
            # with a high mastery score.
            mastery.mastery_score = 90

            # Save the test setup.
            db.commit()

            # Build the complete Tutor Agent graph.
            graph = build_tutor_graph()

            # Run the graph for the test learner.
            result = graph.invoke(
                {
                    "user_id": 2,
                    "session_id": 1,
                    "user_message": (
                        "What should I learn next?"
                    )
                }
            )

            # High mastery should cause the agent
            # to recommend the learner's next step.
            assert (
                result["recommended_action"]
                == "recommend"
            )

            # Because Agentic AI is the final topic
            # in this learning path, the recommendation
            # node should detect path completion.
            assert (
                result["response"]
                == (
                    "You have completed all topics "
                    "in your learning path."
                )
            )

            # Refresh database state because the graph
            # uses separate database sessions.
            db.expire_all()

            # Reload the Agentic AI learning path item
            # after the graph has completed.
            updated_item = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id == 1,
                    LearningPathItem.topic_id == 3
                )
                .first()
            )

            # Verify that the current topic was marked
            # as completed before recommendation.
            assert updated_item is not None
            assert updated_item.status == "completed"

            # Verify that the completed topic now
            # recommends moving to the next step.
            assert (
                updated_item.recommended_action
                == "recommend"
            )

            # Reload the learning path after the graph
            # completes the final topic.
            updated_path = (
                db.query(LearningPath)
                .filter(
                    LearningPath.learning_path_id == 1
                )
                .first()
            )

            # Verify that completing the final topic
            # also completes the entire learning path.
            assert updated_path is not None
            assert updated_path.status == "completed"

            # Verify that the completed path status
            # is also reflected in TutorState.
            assert (
                result["learning_path"]["status"]
                == "completed"
            )

        finally:
            # Roll back any failed transaction
            # before restoring the test data.
            db.rollback()

            # Reload the mastery record because
            # the graph used separate database sessions.
            mastery = (
                db.query(TopicMastery)
                .filter(
                    TopicMastery.user_id == 2,
                    TopicMastery.topic_id == 3
                )
                .first()
            )

            # Reload the learning path item
            # before restoring its original state.
            path_item = (
                db.query(LearningPathItem)
                .filter(
                    LearningPathItem.learning_path_id == 1,
                    LearningPathItem.topic_id == 3
                )
                .first()
            )

            # Reload the learning path because
            # the graph updated it in another session.
            learning_path = (
                db.query(LearningPath)
                .filter(
                    LearningPath.learning_path_id == 1
                )
                .first()
            )

            # Restore the original mastery score.
            if mastery is not None:
                mastery.mastery_score = original_score

            # Restore the original learning path
            # item status and recommended action.
            if path_item is not None:
                path_item.status = original_status
                path_item.recommended_action = (
                    original_action
                )

            # Restore the original learning path status.
            if learning_path is not None:
                learning_path.status = (
                    original_path_status
                )

            # Save all restored test data.
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
    assert "Mastery is 90%" in result["response"]
    assert "Learn Agentic AI" in result["response"]


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
    Assessment inputs should be prepared from
    the current topic and learner context.
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
    }

    result = prepare_assessment_inputs(state)

    assert result["topics"] == [
        {
            "topic_id": 12,
            "topic": "Building Your First Agent",
        }
    ]

    assert result["student_level"] == "intermediate"
    assert result["assessment_type"] == "topic"


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
    }

    result = prepare_assessment_inputs(state)

    assert result == {}


def test_generate_assessment_node_generates_quiz(monkeypatch):
    """
    The assessment node should retrieve learning context,
    generate a quiz, and store the generated questions
    in TutorState.
    """

    # Mock the RAG retrieval step so the test
    # does not require embeddings or an API call.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )
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
        """
        Fake quiz tool used to avoid calling
        the real LLM during testing.
        """

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
    }

    result = generate_assessment_node(state)

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

    # Simulate a RAG retrieval result
    # with no available learning material.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: ""
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
    updated topic mastery, and return the completed
    assessment result.
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
        lambda db, user_id, topic_id,
        assessment_type, questions: FakeAssessmentAttempt()
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

    # Verify the final response shown
    # after assessment submission.
    assert "Assessment completed" in result["response"]
    assert "1/1" in result["response"]


    # Verify that the learner sees the updated
    # mastery and recommended next step.
    assert "Mastery: 100%" in result["response"]
    assert "Recommended next step: recommend" in result["response"]



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
    assert "answers are incomplete" in result["response"]


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



def test_prepare_teaching_inputs():
    """
    Teaching inputs should be prepared from
    the current topic and learner context.
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
    }

    result = prepare_teaching_inputs(state)

    assert result == {
        "topic_id": 12,
        "topic_name": "Building Your First Agent",
        "student_level": "intermediate",
    }


def test_prepare_teaching_inputs_without_topic():
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

    result = prepare_teaching_inputs(state)

    assert result == {}


def test_teach_node_generates_explanation(
    monkeypatch
):
    """
    The teaching node should retrieve topic context
    and generate a personalized explanation.
    """

    # Mock RAG retrieval so the test does not
    # require embeddings or a real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
            "AI agents can reason, use tools, "
            "and perform actions to achieve goals."
        )
    )

    # Create a fake explanation tool so the test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            assert inputs["topic"] == "Building Your First Agent"
            assert inputs["student_level"] == "intermediate"
            assert "AI agents can reason" in inputs["context"]

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
    }

    result = teach_node(state)

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
        lambda topic_id, topic_name: ""
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


def test_prepare_practice_inputs():
    """
    Practice inputs should include the current topic,
    learner level, and detected weak areas.
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
        "topic_mastery": {
            "mastery_score": 55,
            "weak_areas": [
                "Tool selection",
                "Tool arguments",
            ],
        },
    }

    result = prepare_practice_inputs(state)

    assert result == {
        "topic_id": 12,
        "topic_name": "Building Your First Agent",
        "student_level": "intermediate",
        "weak_areas": [
            "Tool selection",
            "Tool arguments",
        ],
    }


def test_prepare_practice_inputs_without_topic():
    """
    Practice input preparation should stop safely
    when no current topic is available.
    """

    state = {
        "current_topic": {},
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

    result = prepare_practice_inputs(state)

    assert result == {}


def test_practice_node_generates_personalized_practice(
    monkeypatch
):
    """
    The practice node should retrieve topic context
    and generate personalized practice activities.
    """

    # Mock RAG retrieval so the test does not
    # require embeddings or a real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
            "AI agents can use tools to interact "
            "with external systems."
        )
    )

    # Create a fake practice tool so the test
    # does not call the real language model.
    class FakePracticeTool:
        def invoke(self, inputs):
            assert inputs["topic"] == "Building Your First Agent"
            assert inputs["student_level"] == "intermediate"
            assert "AI agents can use tools" in inputs["context"]

            # Verify that weak areas are passed
            # to the practice tool as JSON.
            assert json.loads(inputs["weak_areas"]) == [
                "Tool selection",
                "Tool arguments",
            ]

            assert inputs["practice_type"] == "flashcards"
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
    }

    result = practice_node(state)

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
        lambda topic_id, topic_name: ""
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


def test_prepare_review_inputs():
    """
    Review inputs should include the current topic,
    learner level, and detected weak areas.
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
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                "Tool selection",
                "Tool arguments",
            ],
        },
    }

    result = prepare_review_inputs(state)

    assert result == {
        "topic_id": 12,
        "topic_name": "Building Your First Agent",
        "student_level": "intermediate",
        "weak_areas": [
            "Tool selection",
            "Tool arguments",
        ],
    }


def test_prepare_review_inputs_without_topic():
    """
    Review input preparation should stop safely
    when no current topic is available.
    """

    state = {
        "current_topic": {},
        "learner_context": {
            "current_level": "intermediate",
        },
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                "Tool selection",
            ],
        },
    }

    result = prepare_review_inputs(state)

    assert result == {}


def test_review_node_generates_focused_review(
    monkeypatch
):
    """
    The review node should retrieve topic context
    and generate a review focused on weak areas.
    """

    # Mock RAG retrieval so the test does not
    # require embeddings or a real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
            "AI agents can reason, use tools, "
            "and interact with external systems."
        )
    )

    # Create a fake explanation tool so the test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            # Verify that the review request contains
            # the topic and the learner's weak areas.
            assert "Building Your First Agent" in inputs["topic"]
            assert "Tool selection" in inputs["topic"]
            assert "Tool arguments" in inputs["topic"]

            # Verify that the learner level and RAG
            # context are passed correctly.
            assert inputs["student_level"] == "intermediate"
            assert "AI agents can reason" in inputs["context"]

            return (
                "Focused review of tool selection "
                "and tool arguments."
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
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [
                "Tool selection",
                "Tool arguments",
            ],
        },
    }

    result = review_node(state)

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

    # Simulate a topic with no available
    # learning material in the knowledge base.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: ""
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
    }

    result = review_node(state)

    assert "no learning context" in result["response"]


def test_review_node_generates_general_review_without_weak_areas(
    monkeypatch
):
    """
    The review node should generate a general topic review
    when no weak areas are available.
    """

    # Mock RAG retrieval so the test does not
    # require embeddings or a real vector database.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name: (
            "AI agents can reason, use tools, "
            "and interact with external systems."
        )
    )

    # Create a fake explanation tool so the test
    # does not call the real language model.
    class FakeExplainTool:
        def invoke(self, inputs):
            # Verify that the request asks for a general
            # review rather than a weak-area-focused review.
            assert "Building Your First Agent" in inputs["topic"]
            assert "summarize its key concepts" in inputs["topic"]
            assert "weak areas" not in inputs["topic"]

            assert inputs["student_level"] == "intermediate"
            assert "AI agents can reason" in inputs["context"]

            return (
                "General review of the key concepts "
                "for building an AI agent."
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
        "topic_mastery": {
            "mastery_score": 75,
            "weak_areas": [],
        },
    }

    result = review_node(state)

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


def test_generate_initial_diagnostic_without_required_topics():
    """
    Test that no diagnostic quiz is generated
    when the selected path has no prerequisite topics.
    """

    with SessionLocal() as db:
        state = {
            "selected_path": "python",
            "learner_context": {
                "current_level": "beginner"
            }
        }

        result = generate_initial_diagnostic_node(
            state,
            db
        )

        assert result["diagnostic_topics"] == []
        assert result["assessment_type"] == "diagnostic"
        assert result["assessment_questions"] == []

        assert (
            "no prerequisite diagnostic assessment"
            in result["response"].lower()
        )

def test_generate_initial_diagnostic_agentic_ai(monkeypatch):
    """
    Test that the Agentic AI path generates
    a diagnostic quiz for its prerequisite topics.
    """

    fake_questions = [
        {
            "topic_id": 1,
            "question": "What is a Python variable?",
        },
        {
            "topic_id": 2,
            "question": "What is supervised learning?",
        },
    ]

    class FakeQuizTool:
        def invoke(self, inputs):
            # Verify that the diagnostic tool receives
            # the correct assessment configuration.
            assert inputs["assessment_type"] == "diagnostic"

            assert inputs["topics"] == [
                {
                    "topic_id": 1,
                    "topic": "Python Basics",
                },
                {
                    "topic_id": 2,
                    "topic": "Machine Learning Basics",
                },
            ]

            return json.dumps(
                {
                    "questions": fake_questions
                }
            )

    # Prevent the test from calling the real RAG pipeline.
    # Accept retrieval_query because diagnostic retrieval
    # now supports focused semantic search queries.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        lambda topic_id, topic_name, retrieval_query=None:
            f"Context for {topic_name}"
    )

    # Prevent the test from calling the real LLM.
    monkeypatch.setattr(
        "app.agent.graph.generate_quiz",
        FakeQuizTool()
    )

    with SessionLocal() as db:
        state = {
            "selected_path": "agentic_ai",
            "learner_context": {
                "current_level": "beginner"
            }
        }

        result = generate_initial_diagnostic_node(
            state,
            db
        )

    assert result["assessment_type"] == "diagnostic"

    assert result["diagnostic_topics"] == [
        {
            "topic_id": 1,
            "topic": "Python Basics",
        },
        {
            "topic_id": 2,
            "topic": "Machine Learning Basics",
        },
    ]

    assert result["assessment_questions"] == fake_questions
    assert result["response"] == (
        "Your diagnostic assessment is ready."
    )


def test_initial_diagnostic_uses_rag_backed_reference_topics(
    monkeypatch
):
    """
    Test that the initial diagnostic uses the correct
    RAG-backed reference topics and focused retrieval queries.

    Diagnostic Topic 1 (Python) -> RAG Topic 27
    Diagnostic Topic 2 (ML)     -> RAG Topic 26
    """

    retrieved_topic_ids = []
    retrieval_queries = []

    # Capture the topic IDs and retrieval queries sent
    # to the RAG retriever.
    def mock_retrieve_topic_context(
        topic_id,
        topic_name,
        retrieval_query=None
    ):
        retrieved_topic_ids.append(topic_id)
        retrieval_queries.append(retrieval_query)

        return (
            f"Learning material for {topic_name}."
        )

    # Return a valid quiz so the diagnostic node
    # can complete without calling the real LLM.
    class MockGenerateQuiz:
        def invoke(self, inputs):
            return json.dumps(
                {
                    "questions": [
                        {
                            "topic_id": 1,
                            "question_text": (
                                "What is a Python variable?"
                            ),
                            "question_type": "multiple_choice",
                            "difficulty": "beginner",
                            "options": [
                                "A named value",
                                "A database",
                                "A model",
                                "A network",
                            ],
                            "correct_answer": "A named value",
                        }
                    ]
                }
            )

    # Replace the real RAG retrieval function with
    # the test mock.
    monkeypatch.setattr(
        "app.agent.graph.retrieve_topic_context",
        mock_retrieve_topic_context
    )

    # Replace the real quiz generator with
    # the test mock.
    monkeypatch.setattr(
        "app.agent.graph.generate_quiz",
        MockGenerateQuiz()
    )

    # Simulate an Agentic AI learner who needs
    # the initial prerequisite diagnostic.
    state = {
        "selected_path": "agentic_ai",
        "learner_context": {
            "current_level": "beginner"
        },
    }

    # Run the diagnostic generation node.
    with SessionLocal() as db:
        result = generate_initial_diagnostic_node(
            state,
            db
        )

    # ---------------------------------------------------------
    # Verify RAG source mapping
    # ---------------------------------------------------------

    # Python diagnostic Topic 1 must retrieve
    # from the actual Python reference material Topic 27.
    #
    # Machine Learning diagnostic Topic 2 must retrieve
    # from the actual ML reference material Topic 26.
    assert retrieved_topic_ids == [27, 26]

    # ---------------------------------------------------------
    # Verify focused retrieval queries
    # ---------------------------------------------------------

    assert retrieval_queries == [
        (
            "Python fundamentals including variables, data types, "
            "lists, dictionaries, control flow, loops, conditions, "
            "functions, and basic Python behavior"
        ),
        (
            "Machine learning fundamentals including supervised "
            "and unsupervised learning, classification, regression, "
            "clustering, features, labels, model training, "
            "and evaluation"
        ),
    ]

    # ---------------------------------------------------------
    # Verify diagnostic topic identities
    # ---------------------------------------------------------

    # The diagnostic topics must remain 1 and 2 because
    # mastery results are stored against these topics.
    assert [
        topic["topic_id"]
        for topic in result["diagnostic_topics"]
    ] == [1, 2]

    # ---------------------------------------------------------
    # Verify assessment type
    # ---------------------------------------------------------

    assert (
        result["assessment_type"]
        == "diagnostic"
    )

    # ---------------------------------------------------------
    # Verify questions were generated
    # ---------------------------------------------------------

    assert len(
        result["assessment_questions"]
    ) == 1



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
        assert target_topic_id == 25
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


def test_submit_initial_diagnostic_unconfigured_path():
    """
    Test that a learning path without a target topic
    cannot create a personalized learning path yet.
    """

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


def test_route_initial_setup_to_normal_tutor():
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

    result = route_initial_setup(state)

    assert result == "continue_tutor"


def test_route_initial_setup_to_generate_diagnostic():
    """
    Test that selecting a new learning path
    starts the initial diagnostic generation flow.
    """

    state = {
        "user_id": 2,
        "selected_path": "agentic_ai",
    }

    result = route_initial_setup(state)

    assert result == "generate_initial_diagnostic"


def test_route_initial_setup_to_submit_diagnostic():
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

    result = route_initial_setup(state)

    assert result == "submit_initial_diagnostic"


def test_tutor_graph_initial_diagnostic_flow():
    """
    Test that the compiled Tutor Agent graph can route
    a newly selected learning path through initial setup.
    """

    # Build the real compiled LangGraph workflow.
    graph = build_tutor_graph()

    # Python currently has no prerequisite diagnostic topics,
    # so this test does not require RAG or an LLM call.
    state = {
        "user_id": 2,
        "selected_path": "python",
    }

    # Run the actual compiled graph.
    result = graph.invoke(state)

    assert result["selected_path"] == "python"
    assert result["assessment_type"] == "diagnostic"
    assert result["diagnostic_topics"] == []
    assert result["assessment_questions"] == []

    assert (
        "no prerequisite diagnostic assessment"
        in result["response"].lower()
    )