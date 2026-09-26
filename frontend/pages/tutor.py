"""Provides the Tutor Agent chat, explanations, quizzes, and practice UI."""

import streamlit as st

from frontend.api_client import (
    create_chat_session,
    get_chat_messages,
    get_chat_sessions,
    send_chat_message,
)


def render():
    """Render the main Tutor page."""

    st.title("Tutor")

    st.write(
        "Learn with your personalized AI tutor."
    )

    # ---------------------------------------------------------
    # Authentication check
    # ---------------------------------------------------------

    # The Tutor requires an authenticated learner
    # because the Chat API is protected.
    token = st.session_state.get("token")

    if not token:
        st.warning(
            "Please log in before using the Tutor."
        )
        return

    # ---------------------------------------------------------
    # Chat session initialization
    # ---------------------------------------------------------

    if "chat_session_id" not in st.session_state:

        try:
            # Look for existing sessions belonging
            # to the authenticated learner.
            sessions_response = get_chat_sessions(token)

        except Exception as exc:
            st.error(
                f"Could not connect to the Tutor API: {exc}"
            )
            return

        if sessions_response.status_code != 200:
            st.error("Could not load Tutor sessions.")
            st.write(sessions_response.text)
            return

        sessions = sessions_response.json().get(
            "sessions",
            [],
        )

        if sessions:
            # The backend returns newest sessions first,
            # so continue the learner's latest conversation.
            st.session_state.chat_session_id = (
                sessions[0]["session_id"]
            )

        else:
            # Create a session only when this learner
            # does not have an existing session.
            try:
                response = create_chat_session(
                    token=token,
                    session_name="Tutor Session",
                )

            except Exception as exc:
                st.error(
                    f"Could not connect to the Tutor API: {exc}"
                )
                return

            if response.status_code != 200:
                st.error(
                    "Could not start a Tutor session."
                )
                st.write(response.text)
                return

            session_data = response.json()

            st.session_state.chat_session_id = (
                session_data["session_id"]
            )

    # ---------------------------------------------------------
    # Stored chat history
    # ---------------------------------------------------------

    if "tutor_messages" not in st.session_state:

        try:
            messages_response = get_chat_messages(
                token=token,
                session_id=(
                    st.session_state.chat_session_id
                ),
            )

        except Exception as exc:
            st.error(
                f"Could not load chat history: {exc}"
            )
            return

        if messages_response.status_code != 200:
            st.error("Could not load chat history.")
            st.write(messages_response.text)
            return

        st.session_state.tutor_messages = (
            messages_response.json().get(
                "messages",
                [],
            )
        )

    # Display previous messages.
    for message in st.session_state.tutor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------------------------------------------------------
    # Learner message
    # ---------------------------------------------------------

    user_message = st.chat_input(
        "Ask your Tutor a question..."
    )

    if not user_message:
        return

    # Show and store the learner message.
    st.session_state.tutor_messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_message)

    # ---------------------------------------------------------
    # Tutor Agent response
    # ---------------------------------------------------------

    try:
        with st.spinner(
            "Tutor is thinking..."
        ):
            response = send_chat_message(
                token=token,
                session_id=(
                    st.session_state.chat_session_id
                ),
                message=user_message,
            )

    except Exception as exc:
        st.error(
            f"Could not connect to the Tutor Agent: {exc}"
        )
        return

    # Stop if the Tutor API returned an error.
    if response.status_code != 200:
        st.error(
            "The Tutor could not process your message."
        )
        st.write(response.text)
        return

    result = response.json()

    tutor_response = result["response"]
    agent_action = result.get("agent_action")

    # Store the Tutor response locally.
    st.session_state.tutor_messages.append(
        {
            "role": "assistant",
            "content": tutor_response,
        }
    )

    # Display the Tutor response.
    with st.chat_message("assistant"):
        st.markdown(tutor_response)

        # Temporarily display the selected action
        # to verify Agent routing during E2E testing.
        if agent_action:
            st.caption(
                f"Agent action: {agent_action}"
            )