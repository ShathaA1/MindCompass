from datetime import datetime, timezone

from app.database.connection import SessionLocal
from app.database.models import ChatSession, ChatMessage
from app.services.chat_service import get_recent_messages


def test_get_recent_messages():
    """
    Test that recent chat messages are loaded
    in chronological order for the correct session.
    """

    with SessionLocal() as db:
        session = None

        try:
            # Create a temporary chat session
            # for the existing test learner.
            session = ChatSession(
                user_id=2,
                session_name="Memory Test Session",
                started_at=datetime.now(timezone.utc)
            )

            db.add(session)
            db.flush()

            # Add temporary messages to the session.
            first_message = ChatMessage(
                session_id=session.chat_session_id,
                role="user",
                content="Explain RAG."
            )

            second_message = ChatMessage(
                session_id=session.chat_session_id,
                role="assistant",
                content="RAG combines retrieval and generation.",
                agent_action="explain"
            )

            db.add_all([
                first_message,
                second_message
            ])
            db.commit()

            # Load the recent conversation history.
            messages = get_recent_messages(
                db=db,
                session_id=session.chat_session_id,
                limit=10
            )

            # Verify that both messages were loaded.
            assert len(messages) == 2

            # Verify that messages are returned
            # in chronological order.
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Explain RAG."

            assert messages[1]["role"] == "assistant"
            assert (
                messages[1]["content"]
                == "RAG combines retrieval and generation."
            )

        finally:
            # Remove temporary test data so the
            # database remains unchanged after the test.
            if session is not None:
                db.query(ChatMessage).filter(
                    ChatMessage.session_id
                    == session.chat_session_id
                ).delete()

                db.query(ChatSession).filter(
                    ChatSession.chat_session_id
                    == session.chat_session_id
                ).delete()

                db.commit()