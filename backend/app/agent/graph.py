from sqlalchemy.orm import Session
import re
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
    get_topic_id_by_key,
)
from langgraph.graph import StateGraph, START, END

from app.agent.planning import (
    determine_learner_need,
    plan_next_topic,
    recommend_next_action,
    resolve_action,
    route_recommended_action,
)

from app.agent.recommendation import build_recommendation

from app.agent.intent_classifier import classify_topic_scope

from app.database.models import ChatMessage

from app.services.chat_service import get_recent_messages
from app.services.assessment_service import save_assessment_result

import json

from app.tools.quiz_generation import generate_quiz
from app.tools.answer_evaluation import evaluate_answer
from app.tools.explain import explain
from app.tools.practice_generation import generate_practice
from app.tools.feedback_generation import generate_feedback

from app.rag.retrieval import retrieve

#----------------------------------------------------------
# Database / state loading helpers
#----------------------------------------------------------

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
    # empty conversation state values.
    if not session_id:
        return {
            "conversation_history": [],
            "last_action": None,
        }

    # Load the most recent messages
    # from the current chat session.
    conversation_history = get_recent_messages(
        db=db,
        session_id=session_id,
        limit=10
    )


    # Load the most recent completed agent action
    # from the current chat session.
    last_action = get_last_agent_action(
        db=db,
        session_id=session_id
    )

    topic_explanation = get_first_explanation(
            db=db,
            session_id=session_id,
        )

    # Store the recent conversation context
    # and last action in the shared agent state.
    return {
        "conversation_history": conversation_history,
        "last_action": last_action,
        "topic_explanation": topic_explanation,
    }


def get_last_agent_action(
    db: Session,
    session_id: int
) -> str | None:
    """
    Return the most recent agent action
    recorded in the current chat session.
    """

    last_message = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.agent_action.isnot(None)
        )
        .order_by(
            ChatMessage.created_at.desc()
        )
        .first()
    )

    if not last_message:
        return None

    return last_message.agent_action

def get_first_explanation(
    db: Session,
    session_id: int
) -> str | None:
    """
    Return the first Tutor explanation
    stored in the current chat session.
    """

    message = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id,
            ChatMessage.role == "assistant",
            ChatMessage.agent_action == "explain",
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .first()
    )

    if not message:
        return None

    return message.content

#----------------------------------------------------------
# LangGraph DB wrappers
#----------------------------------------------------------

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
    Keep the topic already associated with the current
    chat session, or select the next learning-path topic
    when no session topic has been provided.
    """

    # A topic-based chat session should always remain
    # attached to its original topic.
    current_topic = state.get(
        "current_topic",
        {}
    )

    if current_topic.get("topic_id"):
        return {}

    # Fall back to normal learning-path planning when
    # the conversation is not tied to a specific topic.
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


#----------------------------------------------------------
# Routing
#----------------------------------------------------------


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



def route_topic_scope(
    state: TutorState
) -> str:
    """
    Check whether the learner's request belongs
    to the current topic and route accordingly.
    """

    current_topic = state.get(
        "current_topic",
        {}
    )

    topic_name = current_topic.get(
        "name"
    )

    user_message = state.get(
        "user_message",
        ""
    )

    # Continue normally when topic information
    # is not available for scope validation.
    if not topic_name or not user_message:
        return "continue"

    topic_scope = classify_topic_scope(
        user_message=user_message,
        current_topic_name=topic_name,
    )


    if topic_scope == "out_of_scope":
        return "out_of_scope"

    return "continue"

def out_of_scope_node(
    state: TutorState
) -> dict:
    """
    Tell the learner to stay within
    the current learning topic.
    """


    current_topic = state.get(
        "current_topic",
        {}
    )

    topic_name = current_topic.get(
        "name",
        "the current topic"
    )

    return {
        "response": (
            "That request is outside your current learning topic. "
            f"We're currently working on **{topic_name}**. "
            "You can ask for explanations, examples, practice, "
            "review, or assessment related to this topic."
        ),
    }
#----------------------------------------------------------
# Shared personalization helpers
#----------------------------------------------------------

def prepare_personalized_inputs(
    state: TutorState
) -> dict:
    """
    Prepare the shared learner, topic, mastery,
    preference, conversation, and request data
    used by personalized Tutor actions.
    """

    current_topic = state.get(
        "current_topic",
        {}
    )

    if not current_topic:
        return {}

    topic_id = current_topic.get(
        "topic_id"
    )

    topic_name = current_topic.get(
        "name"
    )

    if not topic_id or not topic_name:
        return {}

    learner_context = state.get(
        "learner_context",
        {}
    )

    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    student_level = (
        learner_context.get("current_level")
        or learner_context.get("initial_level")
        or "beginner"
    )

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "student_level": student_level,
        "mastery_score": topic_mastery.get(
            "mastery_score"
        ),
        "weak_areas": topic_mastery.get(
            "weak_areas",
            []
        ),
        "preferred_format": learner_context.get(
            "preferred_format"
        ),
        "preferred_pace": learner_context.get(
            "preferred_pace"
        ),
        "conversation_history": state.get(
            "conversation_history",
            []
        ),
        "user_message": state.get(
            "user_message",
            ""
        ),
    }


def normalize_weak_areas(
    weak_areas: list | None
) -> list[str]:
    """
    Convert weak-area values into a clean list
    of learner-facing area names.
    """

    normalized_areas = []

    for weak_area in weak_areas or []:
        if isinstance(weak_area, dict):
            area_name = weak_area.get(
                "area"
            )

            if area_name:
                normalized_areas.append(
                    area_name
                )
        else:
            normalized_areas.append(
                str(weak_area)
            )

    return normalized_areas


def format_conversation_history(
    conversation_history: list[dict] | None
) -> str:
    """
    Convert recent conversation history into a
    compact text block for Tutor personalization.
    """

    history_parts = []

    for message in conversation_history or []:
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

    return " | ".join(
        history_parts
    )


def build_personalization_parts(
    inputs: dict,
    request_label: str,
) -> list[str]:
    """
    Build shared personalization instructions using
    learner mastery, preferences, request, and history.
    """

    parts = []

    mastery_score = inputs.get(
        "mastery_score"
    )

    if mastery_score is not None:
        parts.append(
            f"The learner's current mastery score is "
            f"{mastery_score:.0f}%."
        )

    preferred_format = inputs.get(
        "preferred_format"
    )

    if preferred_format:
        parts.append(
            "Preferred learning format: "
            f"{preferred_format}."
        )

    preferred_pace = inputs.get(
        "preferred_pace"
    )

    if preferred_pace:
        parts.append(
            "Preferred learning pace: "
            f"{preferred_pace}."
        )

    user_message = inputs.get(
        "user_message"
    )

    if user_message:
        parts.append(
            f"{request_label}: {user_message}"
        )

    conversation_context = format_conversation_history(
        inputs.get(
            "conversation_history"
        )
    )

    if conversation_context:
        parts.append(
            "Recent conversation context: "
            + conversation_context
        )

    return parts


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



#----------------------------------------------------------
# Diagnostic flow
#----------------------------------------------------------

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
        # Load and validate the selected learning path.
        path_config = AVAILABLE_LEARNING_PATHS.get(
            selected_path
        )

        if not path_config:
            return {
                "response": (
                    "The selected learning path is not supported."
                )
            }

        target_topic_key = path_config.get(
            "target_topic_key"
        )

        if not target_topic_key:
            return {
                "response": (
                    "This learning path is not fully configured yet."
                )
            }

        try:
            target_topic_id = get_topic_id_by_key(
                db=db,
                topic_key=target_topic_key,
            )
        except ValueError as error:
            return {"response": str(error)}

        # Create the learning path immediately when
        # no prerequisite diagnostic is required.
        learning_path = create_learning_path(
            db=db,
            user_id=state["user_id"],
            target_topic_id=target_topic_id,
            path_name=f'{path_config["name"]} Learning Path',
            goal=f'Learn {path_config["name"]}',
        )

        # Select the first available topic so the learner
        # can begin the learning path immediately.
        next_topic = select_next_topic(
            db=db,
            learning_path_id=learning_path[
                "learning_path_id"
            ],
        )

        return {
            "diagnostic_topics": [],
            "assessment_type": "diagnostic",
            "assessment_questions": [],
            "learning_path": learning_path,
            "current_topic": next_topic,
            "next_topic_id": (
                next_topic["topic_id"]
                if next_topic
                else None
            ),
            "response": (
                "No prerequisite diagnostic assessment "
                "is required. Your learning path is ready."
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
        # Resolve diagnostic and curriculum topics from stable keys
    # instead of relying on environment-specific numeric IDs.
    python_diagnostic_topic_id = get_topic_id_by_key(
        db,
        "python_ch0_1",
    )
    ml_diagnostic_topic_id = get_topic_id_by_key(
        db,
        "machine_learning_ch0_1",
    )

    python_curriculum_topic_ids = [
        get_topic_id_by_key(
            db,
            f"python_ch{chapter_number}_1",
        )
        for chapter_number in range(1, 9)
    ]

    ml_curriculum_topic_ids = [
        get_topic_id_by_key(
            db,
            f"machine_learning_ch{chapter_number}_1",
        )
        for chapter_number in range(1, 8)
    ]

    diagnostic_question_distribution = {
        "machine_learning": {
            python_diagnostic_topic_id: 10,
        },
        "agentic_ai": {
            python_diagnostic_topic_id: 5,
            ml_diagnostic_topic_id: 5,
        },
    }

    # Map each diagnostic topic to its RAG-backed
    # curriculum using the resolved database IDs.
    diagnostic_rag_sources = {
        python_diagnostic_topic_id: {
            "topic_ids": python_curriculum_topic_ids,
            "topic_name": "Python Fundamentals",
            "retrieval_query": (
                "Python fundamentals including variables, data types, "
                "operators, strings, conditions, loops, collections, "
                "functions, and basic Python programming concepts"
            ),
        },
        ml_diagnostic_topic_id: {
            "topic_ids": ml_curriculum_topic_ids,
            "topic_name": "Machine Learning Fundamentals",
            "retrieval_query": (
                "Machine learning fundamentals including supervised "
                "learning, unsupervised learning, deep learning, "
                "reinforcement learning, ensemble learning, "
                "classification, regression, clustering, model training, "
                "and evaluation"
            ),
        },
    }

    question_distribution = (
        diagnostic_question_distribution.get(
            selected_path,
            {}
        )
    )

    all_questions = []

    # Generate each diagnostic area separately so the
    # requested question distribution is guaranteed.
    for topic in diagnostic_topics:
        diagnostic_topic_id = topic["topic_id"]

        num_questions = question_distribution.get(
            diagnostic_topic_id,
            0
        )

        # Skip diagnostic areas that are not required
        # for the selected learning path.
        if num_questions <= 0:
            continue

        rag_source = diagnostic_rag_sources.get(
            diagnostic_topic_id
        )

        # Skip safely if no RAG source has been configured
        # for this diagnostic area.
        if not rag_source:
            continue

        # Collect relevant learning material from all
        # curriculum topics belonging to this area.
        topic_context_parts = []

        for rag_topic_id in rag_source["topic_ids"]:
            topic_context = retrieve_topic_context(
                topic_id=rag_topic_id,
                topic_name=rag_source["topic_name"],
                retrieval_query=rag_source[
                    "retrieval_query"
                ],
            )

            if topic_context:
                topic_context_parts.append(
                    topic_context
                )

        combined_topic_context = "\n\n".join(
            topic_context_parts
        )

        # Do not generate questions without grounded
        # learning material.
        if not combined_topic_context:
            continue

        # Generate only the required number of questions
        # for this diagnostic knowledge area.
        quiz_json = generate_quiz.invoke(
            {
                "topics": [topic],
                "context": combined_topic_context,
                "student_level": student_level,
                "num_questions": num_questions,
                "assessment_type": "diagnostic",
            }
        )

        quiz = json.loads(
            quiz_json
        )

        generated_questions = quiz.get(
            "questions",
            []
        )

        all_questions.extend(
            generated_questions
        )

    return {
        "diagnostic_topics": diagnostic_topics,
        "assessment_type": "diagnostic",
        "assessment_questions": all_questions,
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

    target_topic_key = path_config.get(
        "target_topic_key"
    )

    if not target_topic_key:
        return {
            "response": (
                "This learning path is not fully configured yet."
            )
        }

    try:
        target_topic_id = get_topic_id_by_key(
            db=db,
            topic_key=target_topic_key,
        )
    except ValueError as error:
        return {"response": str(error)}

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

#----------------------------------------------------------
# Teaching
#----------------------------------------------------------


def teach_node(
    state: TutorState
) -> dict:
    """
    Generate a personalized teaching response using
    the learner's current topic and conversation context.
    """

    # Prepare the learner and topic information
    # required for the teaching workflow.
    teaching_inputs = prepare_personalized_inputs(
        state
    )

    if not teaching_inputs:
        return {
            "response": (
                "An explanation could not be generated "
                "because no current topic is available."
            )
        }

    user_message = teaching_inputs.get(
        "user_message",
        ""
    )

    conversation_history = state.get(
        "conversation_history",
        []
    )

    previous_explanation = state.get(
        "topic_explanation"
    )

    # A stored explanation means the learner has
    # already received the main topic explanation.
    is_follow_up = bool(
        previous_explanation
    )

    # Build a more meaningful retrieval query.
    # Short follow-ups such as "more" still include
    # the current topic so retrieval remains relevant.
    retrieval_query = (
        f'{teaching_inputs["topic_name"]}: '
        f'{user_message}'
    )

    # Retrieve approved learning material from
    # the current topic.
    context = retrieve_topic_context(
        topic_id=teaching_inputs["topic_id"],
        topic_name=teaching_inputs["topic_name"],
        retrieval_query=retrieval_query,
    )

    if not context:
        return {
            "response": (
                "An explanation could not be generated "
                "because no learning context was found "
                "for the current topic."
            )
        }

    personalization_parts = []

    if is_follow_up:
        # Answer only the learner's latest follow-up
        # instead of restarting the full lesson.
        personalization_parts.append(
            (
                f'Continue the discussion about '
                f'"{teaching_inputs["topic_name"]}". '
                "Answer only the learner's latest request. "
                "Use the previous explanation and recent "
                "conversation to understand short follow-ups. "
                "Do not restart, repeat, or summarize the full "
                "topic unless the learner explicitly asks for it."
            )
        )

        personalization_parts.append(
            (
                "The learner's latest request is: "
                f'"{user_message}".'
            )
        )

        # Include the previous explanation so short
        # follow-ups can be interpreted correctly.
        personalization_parts.append(
            (
                "Previous topic explanation: "
                f"{previous_explanation}"
            )
        )

        # Include recent conversation context when available.
        if conversation_history:
            history_text = format_conversation_history(
                conversation_history
            )

            if history_text:
                personalization_parts.append(
                    (
                        "Recent conversation: "
                        f"{history_text}"
                    )
                )

    else:
        # Generate the main explanation when the learner
        # is entering the topic for the first time.
        personalization_parts.append(
            (
                f'Teach the topic '
                f'"{teaching_inputs["topic_name"]}".'
            )
        )

        personalization_parts.extend(
            build_personalization_parts(
                inputs=teaching_inputs,
                request_label=(
                    "The learner specifically asked"
                ),
            )
        )

    weak_area_names = normalize_weak_areas(
        teaching_inputs.get(
            "weak_areas"
        )
    )

    if weak_area_names:
        personalization_parts.append(
            (
                "Focus especially on these weak areas: "
                + ", ".join(weak_area_names)
                + "."
            )
        )

    personalized_topic = " ".join(
        personalization_parts
    )

    # Generate the explanation or follow-up response
    # using approved course material.
    explanation = explain.invoke({
        "topic": personalized_topic,
        "context": context,
        "student_level": teaching_inputs[
            "student_level"
        ],
    })

    # Recalculate the learner's recommended next
    # action after the teaching interaction.
    recommendation = build_recommendation(
        mastery_score=teaching_inputs.get(
            "mastery_score"
        ),
        weak_areas=teaching_inputs.get(
            "weak_areas"
        ),
        last_action="explain",
    )

    return {
        "response": explanation,

        # Keep the original topic explanation as the
        # stable learning context for later practice,
        # review, and assessment generation.
        "topic_explanation": (
            previous_explanation
            or explanation
        ),

        "last_action": "explain",
        "recommended_action": recommendation[
            "recommended_action"
        ],
        "recommendation_reason": recommendation[
            "recommendation_reason"
        ],
    }

#----------------------------------------------------------
# Assessment
#----------------------------------------------------------

def prepare_assessment_inputs(
    state: TutorState
) -> dict:
    """
    Prepare personalized inputs for assessment generation.
    """

    base_inputs = prepare_personalized_inputs(
        state
    )

    if not base_inputs:
        return {}

    return {
        **base_inputs,
        "assessment_type": (
            state.get("assessment_type")
            or "topic"
        ),
    }




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

    # Use the previously generated topic explanation
    # as the primary context for review generation.
    context = state.get("topic_explanation")

    # Fall back to topic retrieval only when no
    # previous explanation is available.
    if not context:
        context = retrieve_topic_context(
            topic_id=assessment_inputs["topic_id"],
            topic_name=assessment_inputs["topic_name"],
            retrieval_query=assessment_inputs["user_message"],
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

    assessment_topic_parts.extend(
        build_personalization_parts(
            inputs=assessment_inputs,
            request_label="The learner specifically requested",
        )
    )


    weak_area_names = normalize_weak_areas(
        assessment_inputs.get(
            "weak_areas"
        )
    )

    if weak_area_names:
        assessment_topic_parts.append(
            "Include questions that check these weak areas: "
            + ", ".join(weak_area_names)
            + "."
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
            mastery_score=updated_topic_mastery["mastery_score"],
            weak_areas=updated_topic_mastery.get("weak_areas"),
            last_action="assess",
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

        # Convert structured feedback into readable text
        # for both the learner response and database record.
        feedback_lines = []

        summary = feedback.get("summary")

        if summary:
            feedback_lines.append(summary)

        strengths = feedback.get(
            "strengths",
            []
        )

        if strengths:
            feedback_lines.append("")
            feedback_lines.append("Strengths:")

            feedback_lines.extend(
                f"- {strength}"
                for strength in strengths
            )

        feedback_weak_areas = feedback.get(
            "weak_areas",
            []
        )

        if feedback_weak_areas:
            feedback_lines.append("")
            feedback_lines.append("Areas to review:")

            feedback_lines.extend(
                f"- {weak_area}"
                for weak_area in feedback_weak_areas
            )

        feedback_recommendation = feedback.get(
            "recommendation"
        )

        if feedback_recommendation:
            feedback_lines.append("")
            feedback_lines.append("Recommendation:")
            feedback_lines.append(
                feedback_recommendation
            )

        feedback_text = "\n".join(
            feedback_lines
        )

        # Save the generated personalized feedback
        # on the assessment attempt.
        attempt.feedback = feedback_text
        db.commit()


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
        "last_action": "assess",
         "response": (
            f"Assessment completed. "
            f"Score: {assessment_result['score']}/"
            f"{assessment_result['max_score']}. "
            f"Mastery: "
            f"{updated_topic_mastery['mastery_score']:.0f}%."
            f"\n\nFeedback:\n{feedback_text}"
            f"\n\nRecommended next step: "
            f"{recommendation['recommended_action']}."
        ),
    }


#----------------------------------------------------------
# Practice
#----------------------------------------------------------

def select_practice_type(
    topic_name: str,
    mastery_score: float | None,
    preferred_format: str | None,
    user_message: str | None = None,
) -> str:
    """
    Select a simplified practice type using
    the topic domain, learner mastery, and
    preferred learning format.
    """

    topic_name_lower = topic_name.lower()

    # Respect an explicitly requested practice format
    # before applying learner profile preferences.
    user_message_lower = (
        user_message.strip().lower()
        if user_message
        else ""
    )

    if "scenario" in user_message_lower:
        return "scenario"

    if "flashcard" in user_message_lower:
        return "flashcards"

    if (
        "short answer" in user_message_lower
        or "short_answer" in user_message_lower
    ):
        return "short_answer"

    if (
        "coding" in user_message_lower
        or "code practice" in user_message_lower
        or "coding practice" in user_message_lower
    ):
        return "coding"

    # Detect whether the topic is technical
    # and suitable for coding practice.
    is_python_or_ml = (
        "python" in topic_name_lower
        or "machine learning" in topic_name_lower
        or "ml" in topic_name_lower
    )
    # Normalize display labels from the frontend into
    # the canonical values used by the agent.
    preferred_format_key = None

    if preferred_format:
        preferred_format_key = (
            preferred_format
            .strip()
            .lower()
            .replace(" ", "_")
        )
        # Map the legacy onboarding label to its
        # current equivalent for existing profiles.
        if preferred_format_key == "explanations_with_examples":
            preferred_format_key = "detailed_explanations"

    # Respect the learner's preferred format first.
    if preferred_format_key == "concise_explanations":
        return "flashcards"

    if preferred_format_key == "detailed_explanations":
        return "short_answer"

    if preferred_format_key in [
        "guided_practice",
        "practical_examples",
    ]:
        if is_python_or_ml:
            return "coding"

        return "scenario"

    # Fall back to mastery-based practice selection.
    if mastery_score is None:
        return "flashcards"

    if mastery_score < 40:
        return "flashcards"

    if mastery_score < 70:
        return "short_answer"

    # Higher-mastery learners receive
    # application-oriented practice.
    if is_python_or_ml:
        return "coding"

    return "scenario"

def extract_practice_item_count(
    user_message: str,
    default: int = 5,
    maximum: int = 10,
) -> int:
    """
    Extract the requested number of practice items.

    Use the default when the learner does not specify a number,
    and limit large requests to avoid excessive generation.
    """

    if not user_message:
        return default

    normalized_message = user_message.strip().lower()

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    item_names = (
        r"questions?|items?|exercises?|problems?|"
        r"flashcards?|scenarios?"
    )

    # Match numeric requests such as:
    # "3 questions" or "2 short practice questions".
    numeric_match = re.search(
        rf"\b(\d+)\s+(?:[a-z-]+\s+){{0,3}}(?:{item_names})\b",
        normalized_message,
    )

    if numeric_match:
        requested_count = int(numeric_match.group(1))

        if requested_count < 1:
            return default

        return min(requested_count, maximum)

    # Match word-based requests such as:
    # "one question" or "three short practice questions".
    word_pattern = "|".join(number_words)

    word_match = re.search(
        rf"\b({word_pattern})\s+"
        rf"(?:[a-z-]+\s+){{0,3}}(?:{item_names})\b",
        normalized_message,
    )

    if word_match:
        requested_count = number_words[word_match.group(1)]
        return min(requested_count, maximum)

    return default

def practice_node(
    state: TutorState
) -> dict:
    """
    Generate personalized practice using the learner's
    current topic, mastery information, and retrieved context.
    """

    # Prepare the learner, topic, and mastery information
    # required for the practice workflow.
    practice_inputs = prepare_personalized_inputs(state)

    if not practice_inputs:
        return {
            "response": (
                "Practice could not be generated "
                "because no current topic is available."
            )
        }
    # Use identified weak areas to retrieve targeted
    # learning context for personalized practice.
    weak_area_names = normalize_weak_areas(
        practice_inputs.get("weak_areas")
    )

    context = state.get("topic_explanation")

    if weak_area_names:
        retrieval_query = (
            f'{practice_inputs["topic_name"]}. '
            f'Learner request: '
            f'{practice_inputs.get("user_message", "")}. '
            f'Focus practice on these weak areas: '
            f'{", ".join(weak_area_names)}.'
        )

        targeted_context = retrieve_topic_context(
            topic_id=practice_inputs["topic_id"],
            topic_name=practice_inputs["topic_name"],
            retrieval_query=retrieval_query,
        )

        # Prefer context retrieved specifically for the
        # weak areas, while retaining a safe fallback.
        context = targeted_context or context

    # When no weak areas are available, use the learner's
    # current request for general topic retrieval.
    elif not context:
        context = retrieve_topic_context(
            topic_id=practice_inputs["topic_id"],
            topic_name=practice_inputs["topic_name"],
            retrieval_query=practice_inputs["user_message"],
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

    personalization_parts.extend(
        build_personalization_parts(
            inputs=practice_inputs,
            request_label="The learner specifically requested",
        )
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
        user_message=practice_inputs.get(
            "user_message"
        ),
    )
    num_items = extract_practice_item_count(
        practice_inputs["user_message"]
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
        "num_items": num_items,
        "weak_areas": json.dumps(
            practice_inputs["weak_areas"]
        ),
    })

    return {
        "response": practice,
        "last_action": "practice",
    }

#----------------------------------------------------------
# Review
#----------------------------------------------------------


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
    review_inputs = prepare_personalized_inputs(state)

    if not review_inputs:
        return {
            "response": (
                "A review could not be generated "
                "because no current topic is available."
            )
        }

    # Use the previously generated topic explanation
    # as the primary context for practice generation.
    context = state.get("topic_explanation")

    # Fall back to topic retrieval only when no
    # previous explanation is available.
    if not context:
        context = retrieve_topic_context(
            topic_id=review_inputs["topic_id"],
            topic_name=review_inputs["topic_name"],
            retrieval_query=review_inputs["user_message"],
    )


    # Stop safely when neither a previous explanation
    # nor retrieved learning context is available.
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

    review_parts.extend(
        build_personalization_parts(
            inputs=review_inputs,
            request_label="The learner specifically requested",
        )
    )


    weak_area_names = normalize_weak_areas(
        review_inputs.get(
            "weak_areas"
        )
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
        "response": review,
        "last_action": "review",
    }

def recommend_action_node(
    state: TutorState
) -> dict:
    """
    Display the recommended learning action and its reason
    without executing the action or updating the learning path.
    """

    suggested_action = state.get(
        "suggested_action",
        "explain"
    )

    recommendation_reason = state.get(
        "recommendation_reason",
        ""
    )

    topic_mastery = state.get(
        "topic_mastery",
        {}
    )

    weak_areas = topic_mastery.get(
        "weak_areas",
        []
    )

    action_labels = {
        "explain": "Review the explanation",
        "practice": "Practice",
        "assess": "Take an assessment",
        "review": "Review the topic",
        "recommend": "Continue to the next topic",
    }

    action_label = action_labels.get(
        suggested_action,
        suggested_action.replace("_", " ").title()
    )

    response_parts = [
        f"Recommended next action: **{action_label}**."
    ]

    if recommendation_reason:
        response_parts.append(
            f"Why: {recommendation_reason}"
        )

    normalized_weak_areas = []

    for weak_area in weak_areas:
        if isinstance(weak_area, dict):
            area_name = weak_area.get("area")

            if area_name:
                normalized_weak_areas.append(area_name)

        elif weak_area:
            normalized_weak_areas.append(
                str(weak_area)
            )

    if normalized_weak_areas:
        response_parts.append(
            "Areas to focus on:\n"
            + "\n".join(
                f"- {area}"
                for area in normalized_weak_areas
            )
        )
    confirmation_messages = {
        "explain": (
            "Would you like me to explain the topic?"
        ),
        "practice": (
            "Would you like me to start the practice activity?"
        ),
        "assess": (
            "Would you like me to start the assessment?"
        ),
        "review": (
            "Would you like me to start the review?"
        ),
        "recommend": (
            "Would you like to continue to the next topic?"
        ),
    }

    response_parts.append(
        confirmation_messages.get(
            suggested_action,
            "Would you like me to start this activity?"
        )
    )

    return {
        "response": "\n\n".join(response_parts),
        "last_action": "recommend",
    }



#----------------------------------------------------------
# Recommendation / Learning Path
#----------------------------------------------------------

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
        "response": (
            f"Great job! You have completed this topic. "
            f"You are ready to continue to "
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


#----------------------------------------------------------
# build_tutor_graph
#----------------------------------------------------------


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


    workflow.add_node(
        "out_of_scope",
        out_of_scope_node
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

    # Add the main Tutor action nodes for
    # teaching, assessment, practice, and review.
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
    # Add a node that displays a recommendation
    # without executing or changing the learning path.
    workflow.add_node(
        "recommend_action",
        recommend_action_node
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

    workflow.add_conditional_edges(
        "plan_next_topic",
        route_topic_scope,
        {
            "continue": "load_topic_mastery",
            "out_of_scope": "out_of_scope",
        }
    )

    workflow.add_edge(
        "out_of_scope",
        END
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
            "recommend_action": "recommend_action",
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
    "recommend_action",
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
