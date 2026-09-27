from sqlalchemy.orm import Session

from app.agent.state import TutorState
from app.database.connection import SessionLocal
from app.services.learner_service import get_learner_context
from app.core.learning_paths import AVAILABLE_LEARNING_PATHS
from app.services.learning_service import (
    get_active_learning_path,
    get_topic_mastery,
    select_next_topic,
    update_learning_path,
    complete_learning_path,
    get_diagnostic_topics,
    create_learning_path,
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

from app.agent.recommendation import build_recommendation

from app.services.chat_service import get_recent_messages
from app.services.assessment_service import save_assessment_result

import json

from app.tools.quiz_generation import generate_quiz
from app.tools.answer_evaluation import evaluate_answer
from app.tools.explain import explain
from app.tools.practice_generation import generate_practice
from app.tools.feedback_generation import generate_feedback

from app.rag.retrieval import retrieve


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



def route_initial_setup(state: TutorState) -> str:
    """
    Decide whether the learner should enter
    the initial diagnostic flow or continue
    with the normal Tutor Agent workflow.
    """

    # A selected path indicates that the learner
    # is currently setting up a new learning path.
    selected_path = state.get("selected_path")

    if not selected_path:
        return "continue_tutor"

    # If diagnostic questions and learner answers
    # are present, process the submitted diagnostic.
    assessment_questions = state.get(
        "assessment_questions",
        []
    )

    assessment_answers = state.get(
        "assessment_answers",
        []
    )

    if assessment_questions and assessment_answers:
        return "submit_initial_diagnostic"

    # Otherwise, generate the initial diagnostic
    # for the selected learning path.
    return "generate_initial_diagnostic"

def route_tutor_entry(
    state: TutorState
) -> str:
    """
    Decide whether to run initial setup,
    submit an existing topic assessment,
    or continue with the normal Tutor workflow.
    """

    selected_path = state.get(
        "selected_path"
    )

    assessment_questions = state.get(
        "assessment_questions",
        []
    )

    assessment_answers = state.get(
        "assessment_answers",
        []
    )

    # Initial diagnostic flow.
    if selected_path:
        if (
            assessment_questions
            and assessment_answers
        ):
            return "submit_initial_diagnostic"

        return "generate_initial_diagnostic"

    # Existing topic assessment submission.
    if (
        assessment_questions
        and assessment_answers
    ):
        return "submit_assessment"

    # Normal Tutor conversation.
    return "continue_tutor"



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

    # Add the node that generates the initial
    # diagnostic for a newly selected learning path.
    workflow.add_node(
        "generate_initial_diagnostic",
        generate_initial_diagnostic_graph_node
    )

    workflow.add_node(
        "submit_initial_diagnostic",
        submit_initial_diagnostic_graph_node
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

    # Add the node that generates a new assessment
    # for the learner's current topic.
    workflow.add_node(
        "generate_assessment",
        generate_assessment_node
    )

    # Add the node that processes submitted
    # assessment answers and updates mastery.
    workflow.add_node(
        "submit_assessment",
        submit_assessment_node
    )

    # Add a routing node that decides whether
    # to generate or submit an assessment.
    workflow.add_node(
        "assessment_router",
        lambda state: {}
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

    # After loading the learner profile, decide
    # whether to run initial path setup or
    # continue with the normal Tutor Agent flow.
    workflow.add_conditional_edges(
        "load_learner_context",
        route_tutor_entry,
        {
            "generate_initial_diagnostic": (
                "generate_initial_diagnostic"
            ),
            "submit_initial_diagnostic": (
                "submit_initial_diagnostic"
            ),
            "submit_assessment": (
                "submit_assessment"
            ),
            "continue_tutor": (
                "load_conversation_history"
            ),
        }
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
            "assess": "assessment_router",
            "practice": "practice",
            "review": "review",
            "recommend": "update_learning_path",
        }
    )


    # Decide whether the learner needs a new
    # assessment or is submitting existing answers.
    workflow.add_conditional_edges(
        "assessment_router",
        route_assessment,
        {
            "generate_assessment": "generate_assessment",
            "submit_assessment": "submit_assessment",
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

    # End after generating a new assessment.
    # The workflow waits for the learner to answer.
    workflow.add_edge(
        "generate_assessment",
        END
    )

    # End after processing submitted answers
    # and updating the learner's mastery.
    workflow.add_edge(
        "submit_assessment",
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

        # End after generating the initial diagnostic.
    # The workflow waits for the learner to submit answers.
    workflow.add_edge(
        "generate_initial_diagnostic",
        END
    )

    # End after processing the diagnostic
    # and creating the personalized learning path.
    workflow.add_edge(
        "submit_initial_diagnostic",
        END
    )

    # Compile the workflow into an executable graph.
    return workflow.compile()



def prepare_teaching_inputs(
    state: TutorState
) -> dict:
    """
    Prepare learner, topic, mastery, preferences,
    conversation context, and user request information
    required for a personalized explanation.
    """

    # Load the learner's current topic.
    current_topic = state.get(
        "current_topic",
        {}
    )

    # Load learner profile and preferences.
    learner_context = state.get(
        "learner_context",
        {}
    )

    # Load the learner's current mastery
    # information for this topic.
    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    # Load recent conversation history so the
    # explanation can remain context-aware.
    conversation_history = state.get(
        "conversation_history",
        []
    )

    # Load the learner's current request so RAG
    # can retrieve material relevant to the question.
    user_message = state.get(
        "user_message",
        ""
    )

    # Teaching requires an active topic.
    if not current_topic:
        return {}

    topic_id = current_topic.get(
        "topic_id"
    )

    topic_name = current_topic.get(
        "name"
    )

    # Stop safely if the topic information
    # is incomplete.
    if not topic_id or not topic_name:
        return {}

    # Prefer the learner's latest assessed level.
    student_level = (
        learner_context.get("current_level")
        or learner_context.get("initial_level")
        or "beginner"
    )

    # Load current mastery information.
    mastery_score = topic_mastery.get(
        "mastery_score"
    )

    weak_areas = topic_mastery.get(
        "weak_areas",
        []
    )

    # Load the learner's preferred learning format
    # and pace from the learner profile.
    preferred_format = learner_context.get(
        "preferred_format"
    )

    preferred_pace = learner_context.get(
        "preferred_pace"
    )

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "student_level": student_level,
        "mastery_score": mastery_score,
        "weak_areas": weak_areas,
        "preferred_format": preferred_format,
        "preferred_pace": preferred_pace,
        "conversation_history": conversation_history,
        "user_message": user_message,
    }



def teach_node(
    state: TutorState
) -> dict:
    """
    Generate a personalized teaching response using
    the learner's current topic and retrieved context.
    """

    # Prepare the learner and topic information
    # required for the teaching workflow.
    teaching_inputs = prepare_teaching_inputs(state)

    if not teaching_inputs:
        return {
            "response": (
                "An explanation could not be generated "
                "because no current topic is available."
            )
        }

    # Retrieve learning material from the current topic
    # using the learner's specific request as the
    # semantic search query.
    context = retrieve_topic_context(
        topic_id=teaching_inputs["topic_id"],
        topic_name=teaching_inputs["topic_name"],
        retrieval_query=teaching_inputs["user_message"]
    )

    if not context:
        return {
            "response": (
                "An explanation could not be generated "
                "because no learning context was found "
                "for the current topic."
            )
        }

 
    # Build personalization instructions using
    # the learner's current state and preferences.
    personalization_parts = [
        f'Teach the topic "{teaching_inputs["topic_name"]}".'
    ]

    mastery_score = teaching_inputs.get(
        "mastery_score"
    )

    if mastery_score is not None:
        personalization_parts.append(
            f"The learner's current mastery score is "
            f"{mastery_score:.0f}%."
        )

    weak_areas = teaching_inputs.get(
        "weak_areas",
        []
    )

    if weak_areas:
        weak_area_names = []

        for weak_area in weak_areas:
            if isinstance(weak_area, dict):
                area_name = weak_area.get(
                    "area"
                )

                if area_name:
                    weak_area_names.append(
                        area_name
                    )
            else:
                weak_area_names.append(
                    str(weak_area)
                )

        if weak_area_names:
            personalization_parts.append(
                "Focus especially on these weak areas: "
                + ", ".join(weak_area_names)
                + "."
            )

    preferred_format = teaching_inputs.get(
        "preferred_format"
    )

    if preferred_format:
        personalization_parts.append(
            "Preferred learning format: "
            f"{preferred_format}."
        )

    preferred_pace = teaching_inputs.get(
        "preferred_pace"
    )

    if preferred_pace:
        personalization_parts.append(
            "Preferred learning pace: "
            f"{preferred_pace}."
        )


    user_message = teaching_inputs.get(
        "user_message"
    )

    if user_message:
        personalization_parts.append(
            "The learner specifically asked: "
            f"{user_message}"
        )


    # Include recent conversation context so the
    # explanation remains consistent across turns.
    conversation_history = teaching_inputs.get(
        "conversation_history",
        []
    )

    if conversation_history:
        history_parts = []

        for message in conversation_history:
            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if content:
                history_parts.append(
                    f"{role}: {content}"
                )

        if history_parts:
            personalization_parts.append(
                "Recent conversation context: "
                + " | ".join(history_parts)
            )


    personalized_topic = " ".join(
        personalization_parts
    )

    # Generate a personalized explanation using
    # learner mastery, weak areas, preferences,
    # and retrieved course material.
    explanation = explain.invoke({
        "topic": personalized_topic,
        "context": context,
        "student_level": teaching_inputs[
            "student_level"
        ],
    })

    return {
        "response": explanation
    }


def prepare_assessment_inputs(
    state: TutorState
) -> dict:
    """
    Prepare learner, topic, mastery, preferences,
    and request information required for generating
    a personalized assessment.
    """

    # Load the learner's current topic.
    current_topic = state.get(
        "current_topic",
        {}
    )

    # Load learner profile information.
    learner_context = state.get(
        "learner_context",
        {}
    )

    # Load mastery information for the
    # learner's current topic.
    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    # Load the learner's current request so RAG
    # can retrieve assessment-relevant material.
    user_message = state.get(
        "user_message",
        ""
    )

    # Assessment generation requires
    # an active topic.
    if not current_topic:
        return {}

    topic_id = current_topic.get(
        "topic_id"
    )

    topic_name = current_topic.get(
        "name"
    )

    # Stop safely if topic information
    # is incomplete.
    if not topic_id or not topic_name:
        return {}

    # Prefer the learner's latest assessed level.
    student_level = (
        learner_context.get("current_level")
        or learner_context.get("initial_level")
        or "beginner"
    )

    # Load current mastery information.
    mastery_score = topic_mastery.get(
        "mastery_score"
    )

    weak_areas = topic_mastery.get(
        "weak_areas",
        []
    )

    # Load learner preferences.
    preferred_format = learner_context.get(
        "preferred_format"
    )

    preferred_pace = learner_context.get(
        "preferred_pace"
    )

    topics = [
        {
            "topic_id": topic_id,
            "topic": topic_name,
        }
    ]

    return {
        "topics": topics,
        "student_level": student_level,
        "assessment_type": (
            state.get("assessment_type")
            or "topic"
        ),
        "mastery_score": mastery_score,
        "weak_areas": weak_areas,
        "preferred_format": preferred_format,
        "preferred_pace": preferred_pace,
        "user_message": user_message,
    }


def retrieve_topic_context(
    topic_id: int,
    topic_name: str,
    retrieval_query: str | None = None
) -> str:
    """
    Retrieve relevant learning material from the RAG
    knowledge base for assessment generation.
    """

    # Use a focused retrieval query when one is provided.
    # Otherwise, fall back to the topic name.
    query = retrieval_query or topic_name

    # Retrieve learning material only from
    # the selected RAG-backed topic.
    retrieved_chunks = retrieve(
        query=query,
        topic_ids=topic_id,
        top_k=5
    )

    # Stop safely if no relevant learning
    # material was retrieved.
    if not retrieved_chunks:
        return ""

    # Combine the retrieved text chunks into
    # one context string for quiz generation.
    context_parts = [
        chunk["text"]
        for chunk in retrieved_chunks
        if chunk.get("text")
    ]

    return "\n\n".join(context_parts)



def generate_initial_diagnostic_node(
    state: TutorState,
    db: Session
) -> dict:
    """
    Generate the initial diagnostic assessment
    for the learner's selected learning path.
    """

    # Get the learning path selected by the learner.
    selected_path = state.get("selected_path")

    # A learning path must be selected before
    # generating the diagnostic assessment.
    if not selected_path:
        return {
            "response": (
                "Please select a learning path "
                "before starting the diagnostic assessment."
            )
        }

    # Load the prerequisite topics that should
    # be assessed for the selected learning path.
    diagnostic_topics = get_diagnostic_topics(
        db,
        selected_path
    )

    # Some learning paths may not currently require
    # a prerequisite diagnostic assessment.
    if not diagnostic_topics:
        return {
            "diagnostic_topics": [],
            "assessment_type": "diagnostic",
            "assessment_questions": [],
            "response": (
                "No prerequisite diagnostic assessment "
                "is required for this learning path."
            ),
        }

    # Load the learner's current level.
    learner_context = state.get(
        "learner_context",
        {}
    )

    student_level = (
        learner_context.get("current_level")
        or learner_context.get("initial_level")
        or "beginner"
    )

    # Map diagnostic topics to the real RAG-backed
    # learning materials used to generate questions.
    #
    # Topic 1 stores Python diagnostic mastery,
    # while Topic 27 contains the actual Python material.
    #
    # Topic 2 stores Machine Learning diagnostic mastery,
    # while Topic 26 contains the actual ML material.
    diagnostic_rag_sources = {
        1: {
            "topic_id": 27,
            "topic_name": "Python Reference Material",
            "retrieval_query": (
                "Python fundamentals including variables, data types, "
                "lists, dictionaries, control flow, loops, conditions, "
                "functions, and basic Python behavior"
            ),
        },
        2: {
            "topic_id": 26,
            "topic_name": "Machine Learning Reference Material",
            "retrieval_query": (
                "Machine learning fundamentals including supervised "
                "and unsupervised learning, classification, regression, "
                "clustering, features, labels, model training, "
                "and evaluation"
            ),
        },
    }

    # Build RAG context from the real learning materials
    # while preserving the diagnostic topic identities.
    context_parts = []

    for topic in diagnostic_topics:
        diagnostic_topic_id = topic["topic_id"]

        rag_source = diagnostic_rag_sources.get(
            diagnostic_topic_id
        )

        # Skip diagnostic topics that do not have
        # a configured RAG-backed source.
        if not rag_source:
            continue

        topic_context = retrieve_topic_context(
            topic_id=rag_source["topic_id"],
            topic_name=rag_source["topic_name"],
            retrieval_query=rag_source["retrieval_query"]
        )

        if topic_context:
            context_parts.append(
                (
                    f"Diagnostic topic: {topic['topic']}\n"
                    f"Source material:\n{topic_context}"
                )
            )

    context = "\n\n".join(context_parts)

    # Generate one diagnostic quiz covering
    # all required prerequisite topics.
    quiz_json = generate_quiz.invoke(
        {
            "topics": diagnostic_topics,
            "context": context,
            "student_level": student_level,
            "num_questions": 5,
            "assessment_type": "diagnostic",
        }
    )

    # Convert the tool output from JSON text
    # into Python data.
    quiz = json.loads(quiz_json)

    return {
        "diagnostic_topics": diagnostic_topics,
        "assessment_type": "diagnostic",
        "assessment_questions": quiz["questions"],
        "response": (
            "Your diagnostic assessment is ready."
        ),
    }

def generate_initial_diagnostic_graph_node(
    state: TutorState
) -> dict:
    """
    LangGraph wrapper that generates the initial diagnostic
    using its own database session.
    """

    # Open a database session for this graph node.
    with SessionLocal() as db:
        return generate_initial_diagnostic_node(
            state=state,
            db=db
        )

def submit_initial_diagnostic_node(
    state: TutorState,
    db: Session
) -> dict:
    """
    Evaluate the learner's diagnostic answers,
    save topic-level mastery, create the personalized
    learning path, and select the first topic to study.
    """

    # Get the learning path selected by the learner.
    selected_path = state.get("selected_path")

    if not selected_path:
        return {
            "response": (
                "Please select a learning path "
                "before submitting the diagnostic assessment."
            )
        }

    # Load and validate the selected path configuration.
    path_config = AVAILABLE_LEARNING_PATHS.get(
        selected_path
    )

    if not path_config:
        return {
            "response": (
                "The selected learning path is not supported."
            )
        }

    # A target topic is required to create
    # the personalized learning path.
    target_topic_id = path_config.get(
        "target_topic_id"
    )

    if target_topic_id is None:
        return {
            "response": (
                "This learning path is not fully configured yet."
            )
        }

    # Load the diagnostic questions and
    # the learner's submitted answers.
    questions = state.get(
        "assessment_questions",
        []
    )

    answers = state.get(
        "assessment_answers",
        []
    )

    if not questions or not answers:
        return {
            "response": (
                "Diagnostic questions and answers "
                "are required before submission."
            )
        }

    # Make sure every diagnostic question
    # has a corresponding learner answer.
    if len(questions) != len(answers):
        return {
            "response": (
                "Please answer all diagnostic questions "
                "before submitting the assessment."
            )
        }

    # Evaluate every learner answer.
    evaluated_questions = []

    for question, answer in zip(
        questions,
        answers
    ):
        evaluation_json = evaluate_answer.invoke(
            {
                "question": question["question_text"],
                "correct_answer": question["correct_answer"],
                "learner_answer": answer["learner_answer"],
            }
        )

        evaluation = json.loads(
            evaluation_json
        )

        # Keep the original question data and add
        # the learner's answer and evaluation result.
        evaluated_questions.append(
            {
                **question,
                "learner_answer": answer["learner_answer"],
                "is_correct": evaluation["is_correct"],
                "score_awarded": evaluation["score_awarded"],
                "feedback": evaluation["feedback"],
            }
        )

    # Save one multi-topic diagnostic attempt.
    # topic_id is None because the diagnostic
    # can contain questions from multiple topics.
    attempt = save_assessment_result(
        db=db,
        user_id=state["user_id"],
        topic_id=None,
        assessment_type="diagnostic",
        questions=evaluated_questions,
    )

    # The assessment service has now updated
    # TopicMastery for every assessed topic.
    # Create the personalized learning path using
    # those updated mastery scores.
    learning_path = create_learning_path(
        db=db,
        user_id=state["user_id"],
        target_topic_id=target_topic_id,
        path_name=f'{path_config["name"]} Learning Path',
        goal=f'Learn {path_config["name"]}',
    )

    # Select the first incomplete topic whose
    # prerequisites have already been completed.
    next_topic = select_next_topic(
        db=db,
        learning_path_id=learning_path["learning_path_id"],
    )

    return {
        "assessment_result": {
            "assessment_attempt_id": (
                attempt.assessment_attempt_id
            ),
            "score": attempt.score,
            "max_score": attempt.max_score,
        },
        "learning_path": learning_path,
        "current_topic": next_topic,
        "next_topic_id": (
            next_topic["topic_id"]
            if next_topic
            else None
        ),
        "response": (
            "Your diagnostic assessment has been completed "
            "and your personalized learning path is ready."
        ),
    }


def submit_initial_diagnostic_graph_node(
    state: TutorState
) -> dict:
    """
    LangGraph wrapper that processes the submitted diagnostic
    using its own database session.
    """

    # Open a database session for this graph node.
    with SessionLocal() as db:
        return submit_initial_diagnostic_node(
            state=state,
            db=db
        )


def generate_assessment_node(
    state: TutorState
) -> dict:
    """
    Generate a topic assessment using the learner's
    current topic, level, and retrieved learning context.
    """

    # Prepare the learner and topic information
    # required by the quiz generation tool.
    assessment_inputs = prepare_assessment_inputs(
        state
    )

    # Stop safely if assessment inputs
    # could not be prepared.
    if not assessment_inputs:
        return {
            "response": (
                "An assessment could not be generated "
                "because no current topic is available."
            )
        }

    # Get the current topic so its learning
    # material can be retrieved from the RAG system.
    current_topic = state.get(
        "current_topic",
        {}
    )

    topic_id = current_topic["topic_id"]
    topic_name = current_topic["name"]

    # Retrieve learning material from the current topic
    # using the learner's specific assessment request
    # as the semantic search query.
    context = retrieve_topic_context(
        topic_id=topic_id,
        topic_name=topic_name,
        retrieval_query=assessment_inputs[
            "user_message"
        ]
    )

    # Do not generate questions without grounded
    # learning material from the knowledge base.
    if not context:
        return {
            "response": (
                "An assessment could not be generated "
                "because no learning context was found "
                "for the current topic."
            )
        }


    # Build personalized assessment instructions
    # using learner mastery, weak areas, and preferences.
    assessment_topic_parts = [
        f'Assess the topic "{topic_name}".'
    ]

    mastery_score = assessment_inputs.get(
        "mastery_score"
    )

    if mastery_score is not None:
        assessment_topic_parts.append(
            f"The learner's current mastery score is "
            f"{mastery_score:.0f}%."
        )

    weak_areas = assessment_inputs.get(
        "weak_areas",
        []
    )

    weak_area_names = []

    for weak_area in weak_areas:
        if isinstance(weak_area, dict):
            area_name = weak_area.get(
                "area"
            )

            if area_name:
                weak_area_names.append(
                    area_name
                )
        else:
            weak_area_names.append(
                str(weak_area)
            )

    if weak_area_names:
        assessment_topic_parts.append(
            "Include questions that check these weak areas: "
            + ", ".join(weak_area_names)
            + "."
        )

    preferred_format = assessment_inputs.get(
        "preferred_format"
    )

    if preferred_format:
        assessment_topic_parts.append(
            "Preferred learning format: "
            f"{preferred_format}."
        )

    preferred_pace = assessment_inputs.get(
        "preferred_pace"
    )

    if preferred_pace:
        assessment_topic_parts.append(
            "Preferred learning pace: "
            f"{preferred_pace}."
        )

    user_message = assessment_inputs.get(
        "user_message"
    )

    if user_message:
        assessment_topic_parts.append(
            "The learner specifically requested: "
            f"{user_message}"
        )

    personalized_assessment_topic = " ".join(
        assessment_topic_parts
    )

    personalized_topics = [
        {
            "topic_id": topic_id,
            "topic": personalized_assessment_topic,
        }
    ]
    # Generate the quiz using the retrieved course
    # material and learner information.
    quiz_result = generate_quiz.invoke(
        {
            "topics": personalized_topics,
            "context": context,
            "student_level": assessment_inputs[
                "student_level"
            ],
            "num_questions": 5,
            "assessment_type": assessment_inputs[
                "assessment_type"
            ],
        }
    )

    # Convert the JSON string returned by the
    # quiz tool into a Python dictionary.
    quiz = json.loads(quiz_result)

    # Extract the generated questions so they can
    # be stored in the shared TutorState.
    questions = quiz.get(
        "questions",
        []
    )

    return {
        "assessment_type": assessment_inputs[
            "assessment_type"
        ],
        "assessment_questions": questions,
        "response": quiz_result,
    }


def prepare_assessment_responses(
    state: TutorState
) -> list[dict]:
    """
    Combine generated assessment questions with
    the learner's submitted answers.
    """

    # Load the questions previously generated
    # for the learner.
    questions = state.get(
        "assessment_questions",
        []
    )

    # Load the learner's submitted answers.
    answers = state.get(
        "assessment_answers",
        []
    )

    # Stop safely if the assessment is incomplete.
    if not questions or not answers:
        return []

    # The number of answers must match the
    # number of generated assessment questions.
    if len(questions) != len(answers):
        return []

    prepared_responses = []

    # Combine each question with its corresponding
    # learner answer.
    for question, answer in zip(
        questions,
        answers
    ):
        response = {
            **question,
            "learner_answer": answer.get(
                "learner_answer"
            ),
        }

        prepared_responses.append(response)

    return prepared_responses


def evaluate_assessment_responses(
    responses: list[dict]
) -> list[dict]:
    """
    Evaluate each learner response using the
    answer evaluation tool.
    """

    # Stop safely when there are no
    # assessment responses to evaluate.
    if not responses:
        return []

    evaluated_responses = []

    # Evaluate each learner answer separately.
    for response in responses:

        # Call the answer evaluation tool using
        # the question and expected answer.
        evaluation_result = evaluate_answer.invoke(
            {
                "question": response["question_text"],
                "correct_answer": response["correct_answer"],
                "learner_answer": response[
                    "learner_answer"
                ],
            }
        )

        # Convert the JSON response returned by
        # the evaluation tool into Python data.
        evaluation = json.loads(
            evaluation_result
        )

        # Combine the original assessment response
        # with the evaluation result.
        evaluated_response = {
            **response,
            "is_correct": evaluation.get(
                "is_correct",
                False
            ),
            "score_awarded": evaluation.get(
                "score_awarded",
                0
            ),
            "feedback": evaluation.get(
                "feedback"
            ),
        }

        evaluated_responses.append(
            evaluated_response
        )

    return evaluated_responses

def submit_assessment_node(
    state: TutorState
) -> dict:
    """
    Evaluate the learner's submitted assessment answers,
    save the assessment result, update topic mastery,
    and synchronize the current learning path item.
    """

    # Combine the generated questions with
    # the learner's submitted answers.
    prepared_responses = prepare_assessment_responses(
        state
    )

    # Stop safely if the assessment responses
    # are incomplete or unavailable.
    if not prepared_responses:
        return {
            "response": (
                "The assessment could not be submitted "
                "because the answers are incomplete."
            )
        }

    # Evaluate each learner answer using
    # the answer evaluation tool.
    evaluated_responses = evaluate_assessment_responses(
        prepared_responses
    )

    # Get the current topic being assessed.
    current_topic = state.get(
        "current_topic",
        {}
    )

    topic_id = current_topic.get(
        "topic_id"
    )

    if not topic_id:
        return {
            "response": (
                "The assessment could not be submitted "
                "because the current topic is missing."
            )
        }

    # Get the assessment type used
    # for this assessment.
    assessment_type = state.get(
        "assessment_type",
        "topic"
    )

    # Save the completed assessment, update mastery,
    # and synchronize the learner's learning path.
    with SessionLocal() as db:

        attempt = save_assessment_result(
            db=db,
            user_id=state["user_id"],
            topic_id=topic_id,
            assessment_type=assessment_type,
            questions=evaluated_responses,
        )

        # Reload the learner's updated mastery after
        # the assessment result has been saved.
        updated_topic_mastery = get_topic_mastery(
            db=db,
            user_id=state["user_id"],
            topic_id=topic_id
        )

        # Build the next-step recommendation using
        # the learner's updated mastery information.
        recommendation = build_recommendation(
            mastery_score=updated_topic_mastery.get(
                "mastery_score"
            ),
            weak_areas=updated_topic_mastery.get(
                "weak_areas",
                []
            )
        )

        # Prepare the completed assessment results
        # for personalized feedback generation.
        assessment_results_text = json.dumps(
            evaluated_responses,
            ensure_ascii=False,
        )

        # Convert weak areas into a format expected
        # by the feedback generation tool.
        weak_areas_text = json.dumps(
            updated_topic_mastery.get(
                "weak_areas",
                []
            ),
            ensure_ascii=False,
        )

        # Load the learner's current level.
        learner_context = state.get(
            "learner_context",
            {}
        )

        student_level = (
            learner_context.get("current_level")
            or learner_context.get("initial_level")
            or "beginner"
        )

        # Generate personalized feedback based on
        # assessment performance and current weak areas.
        feedback_result = generate_feedback.invoke(
            {
                "assessment_results": assessment_results_text,
                "weak_areas": weak_areas_text,
                "student_level": student_level,
            }
        )

        feedback = json.loads(
            feedback_result
        )

        # Load the learner's active learning path so
        # the current path item can be synchronized
        # with the latest assessment result.
        active_learning_path = get_active_learning_path(
            db=db,
            user_id=state["user_id"]
        )

        updated_path_item = {}

        if active_learning_path:
            learning_path_id = active_learning_path.get(
                "learning_path_id"
            )

            if learning_path_id:
                # Update the current path item using
                # the learner's latest mastery score.
                #
                # For example:
                # mastery below 85 keeps the topic pending,
                # while mastery of 85 or higher completes it.
                updated_path_item = update_learning_path(
                    db=db,
                    user_id=state["user_id"],
                    learning_path_id=learning_path_id,
                    topic_id=topic_id
                )

        # Store only the information needed by
        # the Tutor Agent after submission.
        assessment_result = {
            "assessment_attempt_id": (
                attempt.assessment_attempt_id
            ),
            "score": attempt.score,
            "max_score": attempt.max_score,
        }

    # Keep the current topic synchronized with
    # the updated learning path item when available.
    updated_current_topic = {
        **current_topic,
    }

    if updated_path_item:
        updated_current_topic.update({
            "status": updated_path_item.get(
                "status"
            ),
            "recommended_action": (
                updated_path_item.get(
                    "recommended_action"
                )
            ),
        })

    return {
        "assessment_result": assessment_result,
        "assessment_feedback": feedback,
        "topic_mastery": updated_topic_mastery,
        "current_topic": updated_current_topic,
        "recommended_action": recommendation[
            "recommended_action"
        ],
        "recommendation_reason": recommendation[
            "recommendation_reason"
        ],
        "response": (
            f"Assessment completed. "
            f"Score: {assessment_result['score']}/"
            f"{assessment_result['max_score']}. "
            f"Mastery: "
            f"{updated_topic_mastery['mastery_score']:.0f}%. "
            f"Recommended next step: "
            f"{recommendation['recommended_action']}."
        ),
    }


def route_assessment(
    state: TutorState
) -> str:
    """
    Decide whether the learner needs a new assessment
    or is submitting answers to an existing assessment.
    """

    # Load any assessment questions that were
    # previously generated for the learner.
    questions = state.get(
        "assessment_questions",
        []
    )

    # Load any answers submitted by the learner.
    answers = state.get(
        "assessment_answers",
        []
    )

    # If questions and answers are both available,
    # the learner is submitting an assessment.
    if questions and answers:
        return "submit_assessment"

    # Otherwise, generate a new assessment.
    return "generate_assessment"


def prepare_practice_inputs(
    state: TutorState
) -> dict:
    """
    Prepare learner, topic, mastery, preferences,
    and user request information required for
    personalized practice.
    """

    # Load the learner's current topic.
    current_topic = state.get(
        "current_topic",
        {}
    )

    # Load learner profile information.
    learner_context = state.get(
        "learner_context",
        {}
    )

    # Load mastery information for the
    # learner's current topic.
    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    # Load the learner's current request so RAG
    # can retrieve material relevant to the request.
    user_message = state.get(
        "user_message",
        ""
    )

    # Load recent conversation history so practice
    # can remain consistent with previous tutor turns.
    conversation_history = state.get(
        "conversation_history",
        []
    )

    # Practice requires an active topic.
    if not current_topic:
        return {}

    topic_id = current_topic.get(
        "topic_id"
    )

    topic_name = current_topic.get(
        "name"
    )

    # Stop safely if the topic information
    # is incomplete.
    if not topic_id or not topic_name:
        return {}

    # Prefer the learner's current assessed level.
    # Fall back to the initial self-reported level.
    student_level = (
        learner_context.get("current_level")
        or learner_context.get("initial_level")
        or "beginner"
    )

    # Load learner mastery information.
    mastery_score = topic_mastery.get(
        "mastery_score"
    )

    weak_areas = topic_mastery.get(
        "weak_areas",
        []
    )

    # Load learner preferences from the profile.
    preferred_format = learner_context.get(
        "preferred_format"
    )

    preferred_pace = learner_context.get(
        "preferred_pace"
    )

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "student_level": student_level,
        "mastery_score": mastery_score,
        "weak_areas": weak_areas,
        "preferred_format": preferred_format,
        "preferred_pace": preferred_pace,
        "conversation_history": conversation_history,
        "user_message": user_message,
    }

def select_practice_type(
    topic_name: str,
    mastery_score: float | None,
    preferred_format: str | None,
) -> str:
    """
    Select a personalized practice type using
    the topic domain, learner mastery, and
    preferred learning format.
    """

    topic_name_lower = topic_name.lower()

    # Detect the topic domain.
    is_python_topic = (
        "python" in topic_name_lower
    )

    is_ml_topic = (
        "machine learning" in topic_name_lower
        or "ml" in topic_name_lower
    )

    is_agentic_topic = (
        "agent" in topic_name_lower
        or "agentic" in topic_name_lower
    )

    # Respect the learner's preferred format first.
    if preferred_format == "concise_explanations":
        return "flashcards"

    if preferred_format == "detailed_explanations":
        return "short_answer"

    if preferred_format == "guided_practice":
        if is_python_topic or is_ml_topic:
            return "code_completion"

        return "scenario"

    if preferred_format == "practical_examples":
        if is_python_topic or is_ml_topic:
            if (
                mastery_score is not None
                and mastery_score >= 70
            ):
                return "debugging"

            return "coding"

        if is_agentic_topic:
            return "scenario"

    # Fall back to mastery-based selection.
    if mastery_score is None:
        return "flashcards"

    if mastery_score < 40:
        return "true_false"

    if mastery_score < 55:
        return "fill_blank"

    if mastery_score < 70:
        if is_python_topic or is_ml_topic:
            return "code_completion"

        return "short_answer"

    if mastery_score < 85:
        if is_python_topic or is_ml_topic:
            return "coding"

        if is_agentic_topic:
            return "scenario"

        return "short_answer"

    # Higher-mastery learners receive
    # more application-oriented practice.
    if is_python_topic or is_ml_topic:
        return "debugging"

    if is_agentic_topic:
        return "scenario"

    return "short_answer"



def practice_node(
    state: TutorState
) -> dict:
    """
    Generate personalized practice using the learner's
    current topic, mastery information, and retrieved context.
    """

    # Prepare the learner, topic, and mastery information
    # required for the practice workflow.
    practice_inputs = prepare_practice_inputs(state)

    if not practice_inputs:
        return {
            "response": (
                "Practice could not be generated "
                "because no current topic is available."
            )
        }

    # Retrieve learning material from the current topic
    # using the learner's specific request as the
    # semantic search query.
    context = retrieve_topic_context(
        topic_id=practice_inputs["topic_id"],
        topic_name=practice_inputs["topic_name"],
        retrieval_query=practice_inputs["user_message"]
    )

    if not context:
        return {
            "response": (
                "Practice could not be generated "
                "because no learning context was found "
                "for the current topic."
            )
        }

    # Build personalization instructions using
    # the learner's mastery and profile preferences.
    personalization_parts = [
        f'Practice the topic "{practice_inputs["topic_name"]}".'
    ]

    mastery_score = practice_inputs.get(
        "mastery_score"
    )

    if mastery_score is not None:
        personalization_parts.append(
            f"The learner's current mastery score is "
            f"{mastery_score:.0f}%."
        )

    preferred_format = practice_inputs.get(
        "preferred_format"
    )

    if preferred_format:
        personalization_parts.append(
            "Preferred learning format: "
            f"{preferred_format}."
        )

    preferred_pace = practice_inputs.get(
        "preferred_pace"
    )

    if preferred_pace:
        personalization_parts.append(
            "Preferred learning pace: "
            f"{preferred_pace}."
        )

    user_message = practice_inputs.get(
        "user_message"
    )

    if user_message:
        personalization_parts.append(
            "The learner specifically requested: "
            f"{user_message}"
        )

    # Include recent conversation context so practice
    # can build on what the learner discussed previously.
    conversation_history = practice_inputs.get(
        "conversation_history",
        []
    )

    if conversation_history:
        history_parts = []

        for message in conversation_history:
            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if content:
                history_parts.append(
                    f"{role}: {content}"
                )

        if history_parts:
            personalization_parts.append(
                "Recent conversation context: "
                + " | ".join(history_parts)
            )

    personalized_topic = " ".join(
        personalization_parts
    )


    # Select a personalized practice type using
    # the learner's topic, mastery, and preferences.
    practice_type = select_practice_type(
        topic_name=practice_inputs[
            "topic_name"
        ],
        mastery_score=practice_inputs.get(
            "mastery_score"
        ),
        preferred_format=practice_inputs.get(
            "preferred_format"
        ),
    )
    # Generate personalized practice using the retrieved
    # course material, learner mastery, weak areas,
    # level, and profile preferences.
    practice = generate_practice.invoke({
        "topic": personalized_topic,
        "context": context,
        "student_level": practice_inputs[
            "student_level"
        ],
        "practice_type": practice_type,
        "num_items": 5,
        "weak_areas": json.dumps(
            practice_inputs["weak_areas"]
        ),
    })

    return {
        "response": practice
    }

def prepare_review_inputs(
    state: TutorState
) -> dict:
    """
    Prepare learner, topic, mastery, preferences,
    and user request information required for
    a personalized review.
    """

    # Load the learner's current topic.
    current_topic = state.get(
        "current_topic",
        {}
    )

    # Load learner profile information.
    learner_context = state.get(
        "learner_context",
        {}
    )

    # Load mastery information for the
    # learner's current topic.
    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    # Load the learner's current request so RAG
    # can retrieve material relevant to the review.
    user_message = state.get(
        "user_message",
        ""
    )

    # Load recent conversation history so review
    # can refer back to earlier explanations and questions.
    conversation_history = state.get(
        "conversation_history",
        []
    )

    # Review requires an active topic.
    if not current_topic:
        return {}

    topic_id = current_topic.get(
        "topic_id"
    )

    topic_name = current_topic.get(
        "name"
    )

    # Stop safely if the topic information
    # is incomplete.
    if not topic_id or not topic_name:
        return {}

    # Prefer the learner's current assessed level.
    # Fall back to the initial self-reported level.
    student_level = (
        learner_context.get("current_level")
        or learner_context.get("initial_level")
        or "beginner"
    )

    # Load current mastery information.
    mastery_score = topic_mastery.get(
        "mastery_score"
    )

    weak_areas = topic_mastery.get(
        "weak_areas",
        []
    )

    # Load learner preferences from the profile.
    preferred_format = learner_context.get(
        "preferred_format"
    )

    preferred_pace = learner_context.get(
        "preferred_pace"
    )

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "student_level": student_level,
        "mastery_score": mastery_score,
        "weak_areas": weak_areas,
        "preferred_format": preferred_format,
        "preferred_pace": preferred_pace,
        "conversation_history": conversation_history,
        "user_message": user_message,
    }


def review_node(
    state: TutorState
) -> dict:
    """
    Generate a personalized review using the learner's
    current topic, mastery information, user request,
    and retrieved context.
    """

    # Prepare the learner, topic, mastery information,
    # and current request required for the review.
    review_inputs = prepare_review_inputs(state)

    if not review_inputs:
        return {
            "response": (
                "A review could not be generated "
                "because no current topic is available."
            )
        }

    # Retrieve learning material from the current topic
    # using the learner's specific review request as the
    # semantic search query.
    context = retrieve_topic_context(
        topic_id=review_inputs["topic_id"],
        topic_name=review_inputs["topic_name"],
        retrieval_query=review_inputs["user_message"]
    )

    if not context:
        return {
            "response": (
                "A review could not be generated "
                "because no learning context was found "
                "for the current topic."
            )
        }

    # Build a personalized review request using
    # mastery, weak areas, preferences, and user intent.
    review_parts = [
        f'Review the topic "{review_inputs["topic_name"]}".'
    ]

    mastery_score = review_inputs.get(
        "mastery_score"
    )

    if mastery_score is not None:
        review_parts.append(
            f"The learner's current mastery score is "
            f"{mastery_score:.0f}%."
        )

    weak_areas = review_inputs.get(
        "weak_areas",
        []
    )

    # Normalize weak areas because they may be stored
    # either as strings or structured dictionaries.
    weak_area_names = []

    for weak_area in weak_areas:
        if isinstance(weak_area, dict):
            area_name = weak_area.get(
                "area"
            )

            if area_name:
                weak_area_names.append(
                    area_name
                )
        else:
            weak_area_names.append(
                str(weak_area)
            )

    if weak_area_names:
        review_parts.append(
            "Focus especially on these weak areas: "
            + ", ".join(weak_area_names)
            + "."
        )
    else:
        review_parts.append(
            "Summarize the key concepts of the topic "
            "and reinforce the most important ideas."
        )

    preferred_format = review_inputs.get(
        "preferred_format"
    )

    if preferred_format:
        review_parts.append(
            "Preferred learning format: "
            f"{preferred_format}."
        )

    preferred_pace = review_inputs.get(
        "preferred_pace"
    )

    if preferred_pace:
        review_parts.append(
            "Preferred learning pace: "
            f"{preferred_pace}."
        )

    user_message = review_inputs.get(
        "user_message"
    )

    if user_message:
        review_parts.append(
            "The learner specifically requested: "
            f"{user_message}"
        )


    # Include recent conversation context so review
    # can reinforce material discussed earlier.
    conversation_history = review_inputs.get(
        "conversation_history",
        []
    )

    if conversation_history:
        history_parts = []

        for message in conversation_history:
            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if content:
                history_parts.append(
                    f"{role}: {content}"
                )

        if history_parts:
            review_parts.append(
                "Recent conversation context: "
                + " | ".join(history_parts)
            )


    review_topic = " ".join(
        review_parts
    )

    # Reuse the explanation tool to generate a focused
    # review grounded in the retrieved course material.
    review = explain.invoke({
        "topic": review_topic,
        "context": context,
        "student_level": review_inputs["student_level"],
    })

    return {
        "response": review
    }


def recommend_node(
    state: TutorState
) -> dict:
    """
    Recommend the learner's next topic after completing
    the current topic and provide a personalized reason
    for the recommendation.
    """

    # Get the active learning path
    # from the shared agent state.
    learning_path = state.get(
        "learning_path",
        {}
    )

    # Get learner information that can help
    # personalize the recommendation.
    learner_context = state.get(
        "learner_context",
        {}
    )

    # Get the recommendation reason produced
    # from the learner's latest mastery data.
    recommendation_reason = state.get(
        "recommendation_reason",
        ""
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

        # Select the next incomplete topic whose
        # prerequisites have already been completed.
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

    # Get the learner's goal when available.
    learner_goal = learner_context.get("goal")

    # Build a personalized recommendation response.
    response_parts = [
        f"Your next recommended topic is {next_topic['name']}."
    ]

    # Include the reasoning behind the progression
    # decision when it is available.
    if recommendation_reason:
        response_parts.append(
            recommendation_reason
        )

    # Connect the recommendation to the learner's
    # broader learning goal when one is available.
    if learner_goal:
        response_parts.append(
            f"This topic supports your learning goal: "
            f"{learner_goal}."
        )

    return {
        "current_topic": next_topic,
        "next_topic_id": next_topic["topic_id"],
        "response": " ".join(response_parts),
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



