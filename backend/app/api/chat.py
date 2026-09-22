"""Provides API endpoints for interacting with the Tutor Agent."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.graph import build_tutor_graph
from app.core.security import get_current_user
from app.database.connection import get_db
from app.database.models import ChatSession
from app.schemas.chat import (
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
    Create a new Tutor chat session for
    the authenticated learner.
    """

    # Get the authenticated learner ID
    # from the access token.
    user_id = int(current_user["sub"])

    # Create and persist a new chat session.
    session = create_chat_session(
        db=db,
        user_id=user_id,
        session_name=data.session_name,
    )

    # Return the information needed by the
    # frontend to continue the conversation.
    return {
        "session_id": session.chat_session_id,
        "session_name": session.session_name,
        "started_at": session.started_at,
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
            detail="Tutor Agent did not generate a response.",
        )

    # Read the final action selected by the agent.
    agent_action = result.get(
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
    }