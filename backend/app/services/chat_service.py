from sqlalchemy.orm import Session

from app.database.models import (
    ChatMessage,
    ChatSession,
    utc_now,
)


def get_recent_messages(
    db: Session,
    session_id: int,
    limit: int = 10
) -> list[dict]:
    """
    Load the most recent messages from a chat session
    to provide short-term conversation context.
    """

    # Query the most recent messages first.
    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id
        )
        .order_by(
            ChatMessage.created_at.desc()
        )
        .limit(limit)
        .all()
    )

    # Reverse the result so the conversation
    # is returned in chronological order.
    messages.reverse()

    # Return only the fields needed by the agent.
    return [
        {
            "role": message.role,
            "content": message.content,
            "agent_action": message.agent_action,
        }
        for message in messages
    ]


def create_chat_session(
    db: Session,
    user_id: int,
    session_name: str,
    learning_path_id: int | None = None,
    topic_id: int | None = None,
) -> ChatSession:
    """
    Create a new chat session for the learner.
    """

    # Create the chat session using the learner,
    # learning path, and topic information available.
    session = ChatSession(
        user_id=user_id,
        learning_path_id=learning_path_id,
        topic_id=topic_id,
        session_name=session_name,
        started_at=utc_now(),
    )

    # Persist the new session in the database.
    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def save_chat_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
    agent_action: str | None = None,
    retrieved_sources: list | dict | None = None,
) -> ChatMessage:
    """
    Save a learner or Tutor Agent message
    in an existing chat session.
    """

    # Create the message record.
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        agent_action=agent_action,
        retrieved_sources=retrieved_sources,
    )

    # Persist the message in the database.
    db.add(message)
    db.commit()
    db.refresh(message)

    return message