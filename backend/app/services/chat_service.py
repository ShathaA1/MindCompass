from sqlalchemy.orm import Session

from app.database.models import ChatMessage


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
        }
        for message in messages
    ]