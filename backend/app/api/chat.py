"""Provides API endpoints for interacting with the Tutor Agent."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.graph import build_tutor_graph
from app.core.security import get_current_user
from app.database.connection import get_db
from app.database.models import ChatMessage, ChatSession
from app.schemas.chat import (
    AssessmentSubmissionRequest,
    ChatMessageRequest,
    ChatSessionCreate,
)
from app.services.chat_service import (
    create_chat_session,
    save_chat_message,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("/sessions")
def start_chat_session(
    data: ChatSessionCreate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new Tutor chat session and automatically
    generate the first explanation for a topic session.
    """

    # Get the authenticated learner ID
    # from the access token.
    user_id = int(current_user["sub"])

    # Create and persist the new topic session.
    session = create_chat_session(
        db=db,
        user_id=user_id,
        session_name=data.session_name,
        learning_path_id=data.learning_path_id,
        topic_id=data.topic_id,
    )

    initial_response = None
    recommended_action = None

    # Automatically start learning when this is
    # a topic-based Tutor session.
    if data.topic_id is not None:
        graph = build_tutor_graph()

        initial_state = {
            "user_id": user_id,
            "session_id": session.chat_session_id,

            # Keep the new session attached to
            # the topic selected by the learner.
            "current_topic": (
                {
                    "topic_id": session.topic.topic_id,
                    "name": session.topic.name,
                    "description": session.topic.description,
                    "difficulty_level": (
                        session.topic.difficulty_level
                    ),
                }
                if session.topic
                else {}
            ),

            # This is an internal instruction used only
            # to start the Tutor workflow.
            "user_message": (
                "Explain the current topic clearly "
                "and introduce its main concepts."
            ),
        }

        try:
            result = graph.invoke(
                initial_state
            )

            initial_response = result.get(
                "response"
            )

            recommended_action = result.get(
                "recommended_action"
            )

            agent_action = result.get(
                "last_action"
            )

            # Save only the Tutor explanation.
            # No artificial learner message is stored.
            if (
                initial_response
                and agent_action == "explain"
            ):
                save_chat_message(
                    db=db,
                    session_id=session.chat_session_id,
                    role="assistant",
                    content=initial_response,
                    agent_action="explain",
                )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    # Return the session together with the
    # automatically generated first explanation.
    return {
        "session_id": session.chat_session_id,
        "session_name": session.session_name,
        "learning_path_id": session.learning_path_id,
        "topic_id": session.topic_id,
        "started_at": session.started_at,
        "initial_response": initial_response,
        "recommended_action": recommended_action,
    }

@router.get("/sessions")
def get_chat_sessions(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return chat sessions belonging to
    the authenticated learner.
    """

    user_id = int(current_user["sub"])

    sessions = (
        db.query(ChatSession)
        .filter(
            ChatSession.user_id == user_id
        )
        .order_by(
            ChatSession.started_at.desc()
        )
        .all()
    )

    return {
        "sessions": [
            {
                "session_id": session.chat_session_id,
                "session_name": session.session_name,
                "learning_path_id": session.learning_path_id,
                "topic_id": session.topic_id,
                "started_at": session.started_at,
                "ended_at": session.ended_at,
            }
            for session in sessions
        ]
    }


@router.get("/{session_id}/messages")
def get_chat_messages(
    session_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return stored messages from a chat session
    belonging to the authenticated learner.
    """

    user_id = int(current_user["sub"])

    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.chat_session_id == session_id,
            ChatSession.user_id == user_id,
        )
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id
        )
        .order_by(
            ChatMessage.created_at.asc(),
            ChatMessage.chat_message_id.asc(),
        )
        .all()
    )

    return {
        "session_id": session.chat_session_id,
        "session_name": session.session_name,
        "messages": [
            {
                "message_id": message.chat_message_id,
                "role": message.role,
                "content": message.content,
                "agent_action": message.agent_action,
                "created_at": message.created_at,
            }
            for message in messages
        ],
    }


@router.post("/{session_id}/messages")
def send_chat_message(
    session_id: int,
    data: ChatMessageRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Send a learner message to the Tutor Agent
    and return the generated response.
    """

    # Get the authenticated learner ID
    # from the access token.
    user_id = int(current_user["sub"])

    # Load the requested chat session.
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.chat_session_id == session_id,
            ChatSession.user_id == user_id,
        )
        .first()
    )

    # Prevent access to missing sessions or
    # sessions that belong to another learner.
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    # Build the Tutor Agent workflow.
    graph = build_tutor_graph()

    # Pass the current learner message separately
    # from the stored conversation history.
    initial_state = {
        "user_id": user_id,
        "session_id": session_id,
        "user_message": data.message,

        # Keep the Tutor tied to the topic that belongs
        # to this specific chat session.
        "current_topic": (
            {
                "topic_id": session.topic.topic_id,
                "name": session.topic.name,
                "description": session.topic.description,
                "difficulty_level": (
                    session.topic.difficulty_level
                ),
            }
            if session.topic
            else {}
        ),
    }

    try:
        # Run the Tutor Agent workflow.
        result = graph.invoke(initial_state)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Read the final Tutor response generated
    # by the executed action node.
    response = result.get("response")

    if not response:
        raise HTTPException(
            status_code=500,
            detail=(
                "Tutor Agent did not generate "
                "a response."
            ),
        )

    # Detect assessment generation separately because
    # generating a quiz does not count as a completed
    # assessment action yet.
    assessment_questions = result.get(
        "assessment_questions",
        []
    )

    if assessment_questions:
        agent_action = "assess"
    else:
        agent_action = result.get(
            "last_action"
        )

    # Read the next action recommended
    # for the learner after this interaction.
    recommended_action = result.get(
        "recommended_action"
    )

    # Save the learner message only after the graph
    # has successfully completed. This prevents the
    # current message from appearing twice in context.
    save_chat_message(
        db=db,
        session_id=session_id,
        role="user",
        content=data.message,
    )

    # Save the Tutor Agent response together with
    # the action that produced it.
    save_chat_message(
        db=db,
        session_id=session_id,
        role="assistant",
        content=response,
        agent_action=agent_action,
    )

    # Return the information required by
    # the Tutor frontend.
    return {
        "session_id": session_id,
        "response": response,
        "agent_action": agent_action,
        "recommended_action": recommended_action,
        "next_topic_id": result.get(
            "next_topic_id"
        ),
        "assessment_questions": assessment_questions,
        "assessment_type": result.get(
            "assessment_type"
        ),
    }


@router.post("/{session_id}/assessment")
def submit_chat_assessment(
    session_id: int,
    data: AssessmentSubmissionRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Submit answers for a Tutor assessment and
    return the updated mastery and recommendation.
    """

    # Get the authenticated learner ID.
    user_id = int(current_user["sub"])

    # Verify that the chat session exists and
    # belongs to the authenticated learner.
    session = (
        db.query(ChatSession)
        .filter(
            ChatSession.chat_session_id == session_id,
            ChatSession.user_id == user_id,
        )
        .first()
    )

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    # An assessment submission must contain
    # the generated assessment questions.
    if not data.assessment_questions:
        raise HTTPException(
            status_code=400,
            detail="Assessment questions are missing.",
        )

    # Read the assessed topic from the generated
    # questions. Topic assessments are generated
    # for one current topic.
    topic_id = data.assessment_questions[0].get(
        "topic_id"
    )

    if not topic_id:
        raise HTTPException(
            status_code=400,
            detail="Assessment topic is missing.",
        )

    # Build the Tutor Agent workflow.
    graph = build_tutor_graph()

    # Restore the assessment state required by
    # submit_assessment_node. The topic ID comes
    # from the assessment that was generated for
    # the learner.
    initial_state = {
        "user_id": user_id,
        "session_id": session_id,
        "assessment_type": data.assessment_type,
        "assessment_questions": (
            data.assessment_questions
        ),
        "assessment_answers": [
            answer.model_dump()
            for answer in data.assessment_answers
        ],
        "current_topic": {
            "topic_id": topic_id,
        },
    }

    try:
        # Run the Tutor Agent workflow.
        result = graph.invoke(initial_state)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Read the assessment result response.
    response = result.get("response")

    if not response:
        raise HTTPException(
            status_code=500,
            detail=(
                "Tutor Agent did not generate "
                "an assessment result."
            ),
        )

    # Return the completed assessment information
    # required by the Tutor frontend.
    return {
        "session_id": session_id,
        "response": response,
        "assessment_result": result.get(
            "assessment_result"
        ),
        "topic_mastery": result.get(
            "topic_mastery"
        ),
        "recommended_action": result.get(
            "recommended_action"
        ),
        "recommendation_reason": result.get(
            "recommendation_reason"
        ),
    }