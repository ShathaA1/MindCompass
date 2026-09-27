"""Provides the Tutor Agent chat, explanations, quizzes, and practice UI."""

import streamlit as st

from frontend.api_client import (
    create_chat_session,
    get_chat_messages,
    get_chat_sessions,
    send_chat_message,
    submit_chat_assessment,
)


def render_assessment(
    token,
    session_id,
):
    """
    Render the active Tutor assessment and allow
    the learner to submit selected answers.
    """

    assessment = st.session_state.get(
        "active_assessment"
    )

    if not assessment:
        return

    questions = assessment.get(
        "questions",
        []
    )

    if not questions:
        return

    st.subheader("Assessment")

    st.write(
        "Answer the questions below, then submit "
        "your assessment."
    )

    # Keep all assessment controls inside one form
    # so selecting an option does not trigger
    # unnecessary Streamlit reruns.
    with st.form("tutor_assessment_form"):

        selected_answers = []

        for index, question in enumerate(
            questions
        ):
            question_text = question.get(
                "question_text",
                f"Question {index + 1}",
            )

            options = question.get(
                "options",
                [],
            )

            st.markdown(
                f"**{index + 1}. {question_text}**"
            )

            # Use None as the initial selection so
            # unanswered questions remain detectable.
            selected_answer = st.radio(
                "Choose an answer:",
                options,
                index=None,
                key=f"assessment_question_{index}",
            )

            selected_answers.append(
                selected_answer
            )

        submitted = st.form_submit_button(
            "Submit Assessment"
        )

    if not submitted:
        return

    # Require an answer for every question.
    if any(
        answer is None
        for answer in selected_answers
    ):
        st.warning(
            "Please answer all questions before submitting."
        )
        return

    assessment_answers = [
        {
            "learner_answer": answer,
        }
        for answer in selected_answers
    ]

    try:
        with st.spinner(
            "Evaluating your assessment..."
        ):
            response = submit_chat_assessment(
                token=token,
                session_id=session_id,
                assessment_type=assessment.get(
                    "assessment_type",
                    "topic",
                ),
                assessment_questions=questions,
                assessment_answers=assessment_answers,
            )

    except Exception as exc:
        st.error(
            f"Could not submit the assessment: {exc}"
        )
        return

    if response.status_code != 200:
        st.error(
            "The Tutor could not submit the assessment."
        )
        st.write(response.text)
        return

    result = response.json()

    # The assessment is complete, so remove it
    # from the active Streamlit state.
    st.session_state.pop(
        "active_assessment",
        None,
    )

    # Clear radio-button state before another
    # assessment is generated later.
    for index in range(len(questions)):
        st.session_state.pop(
            f"assessment_question_{index}",
            None,
        )

    tutor_response = result.get(
        "response",
        "Assessment completed.",
    )

    # Keep the result visible in the current
    # Tutor conversation.
    st.session_state.tutor_messages.append(
        {
            "role": "assistant",
            "content": tutor_response,
        }
    )

    st.success(tutor_response)

    recommendation_reason = result.get(
        "recommendation_reason"
    )

    if recommendation_reason:
        st.info(recommendation_reason)

    st.rerun()


def render():
    """Render the main Tutor page."""

    st.title("Tutor")

    st.write(
        "Learn with your personalized AI tutor."
    )

    # ---------------------------------------------------------
    # Authentication check
    # ---------------------------------------------------------

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
            sessions_response = get_chat_sessions(
                token
            )

        except Exception as exc:
            st.error(
                f"Could not connect to the Tutor API: {exc}"
            )
            return

        if sessions_response.status_code != 200:
            st.error(
                "Could not load Tutor sessions."
            )
            st.write(
                sessions_response.text
            )
            return

        sessions = sessions_response.json().get(
            "sessions",
            [],
        )

        if sessions:
            # Continue the learner's latest session.
            st.session_state.chat_session_id = (
                sessions[0]["session_id"]
            )

        else:
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
                st.write(
                    response.text
                )
                return

            session_data = response.json()

            st.session_state.chat_session_id = (
                session_data["session_id"]
            )

    session_id = (
        st.session_state.chat_session_id
    )

    # ---------------------------------------------------------
    # Stored chat history
    # ---------------------------------------------------------

    if "tutor_messages" not in st.session_state:

        try:
            messages_response = get_chat_messages(
                token=token,
                session_id=session_id,
            )

        except Exception as exc:
            st.error(
                f"Could not load chat history: {exc}"
            )
            return

        if messages_response.status_code != 200:
            st.error(
                "Could not load chat history."
            )
            st.write(
                messages_response.text
            )
            return

        st.session_state.tutor_messages = (
            messages_response.json().get(
                "messages",
                [],
            )
        )

    # Display previous chat messages.
    for message in st.session_state.tutor_messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    # ---------------------------------------------------------
    # Active assessment
    # ---------------------------------------------------------

    # If an assessment has already been generated,
    # keep it visible across Streamlit reruns.
    if st.session_state.get(
        "active_assessment"
    ):
        render_assessment(
            token=token,
            session_id=session_id,
        )
        return

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
                session_id=session_id,
                message=user_message,
            )

    except Exception as exc:
        st.error(
            f"Could not connect to the Tutor Agent: {exc}"
        )
        return

    if response.status_code != 200:
        st.error(
            "The Tutor could not process your message."
        )
        st.write(
            response.text
        )
        return

    result = response.json()

    tutor_response = result["response"]

    agent_action = result.get(
        "agent_action"
    )

    assessment_questions = result.get(
        "assessment_questions",
        [],
    )

    # ---------------------------------------------------------
    # Assessment response
    # ---------------------------------------------------------

    if (
        agent_action == "assess"
        and assessment_questions
    ):
        # Do not display the raw quiz JSON or expose
        # correct_answer values to the learner.
        st.session_state.active_assessment = {
            "assessment_type": result.get(
                "assessment_type"
            )
            or "topic",
            "questions": assessment_questions,
        }

        # Rerun so the generated assessment is
        # rendered using the MCQ interface above.
        st.rerun()

    # ---------------------------------------------------------
    # Normal Tutor response
    # ---------------------------------------------------------

    st.session_state.tutor_messages.append(
        {
            "role": "assistant",
            "content": tutor_response,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(
            tutor_response
        )

        # Temporary E2E routing information.
        if agent_action:
            st.caption(
                f"Agent action: {agent_action}"
            )