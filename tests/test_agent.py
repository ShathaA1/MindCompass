from app.database.connection import SessionLocal
from app.agent.graph import (
    load_learner_context,
    load_active_learning_path,
    load_topic_mastery,
    build_tutor_graph,
)
from app.agent.planning import (
    determine_learner_need,
    route_action,
    plan_next_topic,
    recommend_next_action,
    resolve_action,
    route_recommended_action,
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

    # Strong mastery should lead to assessment
    assert result["recommended_action"] == "assess"



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

    # Simulate a learner who explicitly asks for assessment
    state = {
        "learner_need": "assess",
        "recommended_action": "explain",
    }

    # Resolve the final agent action
    result = resolve_action(state)

    # The explicit learner request should be used
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



def test_tutor_graph_end_to_end():
    """
    Test the Tutor Agent workflow from START to END
    using real learner and learning path data.
    """

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


def test_tutor_graph_routes_to_teach():
    """
    Test that the complete Tutor Agent workflow
    routes the learner to the teaching node.
    """

    # Build the compiled Tutor Agent workflow.
    graph = build_tutor_graph()

    # Create the initial TutorState.
    initial_state = {
        "user_id": 2,
        "user_message": "What should I learn next?"
    }

    # Run the complete workflow.
    result = graph.invoke(initial_state)

    # Verify that the learner asked
    # for a recommendation.
    assert result["learner_need"] == "recommend"

    # Verify that Agentic AI
    # was selected as the current topic.
    assert result["current_topic"]["topic_id"] == 3
    assert result["current_topic"]["name"] == "Introduction to Agentic AI"

    # Verify the learner's current mastery.
    assert result["topic_mastery"]["mastery_score"] == 30

    # A mastery score of 30 should
    # produce an explanation recommendation.
    assert result["recommended_action"] == "explain"

    # The explanation recommendation should
    # route the workflow to the teaching node.
    assert result["response"] == "Teaching action selected."