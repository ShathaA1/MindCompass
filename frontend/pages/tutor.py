"""Provides the Tutor Agent chat, explanations, quizzes, and practice UI."""

import streamlit as st
import json

from frontend.api_client import (
    create_chat_session,
    get_chat_messages,
    get_chat_sessions,
    get_dashboard,
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

    # Load structured personalized feedback returned
    # by the Tutor Agent after assessment submission.
    assessment_feedback = result.get(
        "assessment_feedback",
        {}
    )

    recommendation_reason = result.get(
        "recommendation_reason"
    )

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
            
    # Build a learner-friendly feedback message that
    # remains visible in the Tutor conversation.
    feedback_parts = [
        tutor_response
    ]

    if assessment_feedback:
        summary = assessment_feedback.get(
            "summary"
        )

        strengths = assessment_feedback.get(
            "strengths",
            []
        )

        weak_areas = assessment_feedback.get(
            "weak_areas",
            []
        )

        feedback_recommendation = assessment_feedback.get(
            "recommendation"
        )

        if summary:
            feedback_parts.append(
                f"\n\n**Feedback:** {summary}"
            )

        if strengths:
            feedback_parts.append(
                "\n\n**Strengths:**\n"
                + "\n".join(
                    f"- {strength}"
                    for strength in strengths
                )
            )

        if weak_areas:
            feedback_parts.append(
                "\n\n**Areas to Improve:**\n"
                + "\n".join(
                    f"- {area}"
                    for area in weak_areas
                )
            )

        if feedback_recommendation:
            feedback_parts.append(
                "\n\n**Recommendation:** "
                f"{feedback_recommendation}"
            )

    if recommendation_reason:
        feedback_parts.append(
            "\n\n**Why this next step?** "
            f"{recommendation_reason}"
        )

    complete_feedback = "".join(
        feedback_parts
    )

    # Save the complete feedback in conversation state
    # so it remains visible after Streamlit reruns.
    st.session_state.tutor_messages.append(
        {
            "role": "assistant",
            "content": complete_feedback,
        }
    )

    st.success(
        "Assessment completed."
    )

    st.markdown(
        complete_feedback
    )

    st.rerun()


def render():
    """Render the main Tutor page."""


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
    # Current learning context
    # ---------------------------------------------------------

    try:
        dashboard_response = get_dashboard(
            token
        )

    except Exception as exc:
        st.error(
            f"Could not load learning context: {exc}"
        )
        return

    if dashboard_response.status_code != 200:
        st.error(
            "Could not load learning context."
        )
        return

    dashboard_data = dashboard_response.json()

    current_topic = (
        dashboard_data.get("current_topic")
        or {}
    )

    current_topic_id = current_topic.get(
        "topic_id"
    )

    current_topic_name = current_topic.get(
        "topic_name",
        "No active topic"
    )

    recommended_action = (
        dashboard_data.get("recommended_action")
        or "Not available"
    )

    mastery_score = None

    for topic_mastery in dashboard_data.get(
        "topic_masteries",
        []
    ):
        if topic_mastery.get(
            "topic_id"
        ) == current_topic_id:
            mastery_score = topic_mastery.get(
                "mastery_score"
            )
            break

    mastery_text = (
        f"{mastery_score:.0f}%"
        if mastery_score is not None
        else "Not assessed"
    )

    recommended_action_text = (
        recommended_action
        .replace("_", " ")
        .title()
    )

    # ---------------------------------------------------------
    # Tutor header
    # ---------------------------------------------------------

    st.markdown(
        """
        <style>
        .st-key-tutor_header {
            position: sticky;
            top: 3.5rem;
            z-index: 999;
            background-color: #F7FBFF;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid #DCE6F1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(
        key="tutor_header"
    ):
        st.title("Tutor")

        st.write(
            "Learn with your personalized AI tutor."
        )

        col1, col2, col3 = st.columns(
            [2.5, 1, 1.4]
        )

        with col1:
            st.caption("Current Topic")
            st.markdown(
                f"**{current_topic_name}**"
            )

        with col2:
            st.caption("Mastery")
            st.markdown(
                f"**{mastery_text}**"
            )

        with col3:
            st.caption("Recommended Action")
            st.markdown(
                f"**{recommended_action_text}**"
            )
    
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

    # Parse structured practice data so Streamlit
    # can render it using native UI components.
    practice_data = None

    if agent_action == "practice":
        try:
            practice_data = json.loads(
                tutor_response
            )
        except json.JSONDecodeError:
            practice_data = None

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
    # Practice response
    # ---------------------------------------------------------

    if (
        agent_action == "practice"
        and practice_data
    ):
        practice_type = practice_data.get(
            "practice_type",
            "practice"
        )

        practice_items = practice_data.get(
            "items",
            []
        )

        with st.chat_message("assistant"):
            st.markdown(
                "### "
                + practice_type.replace(
                    "_",
                    " "
                ).title()
            )

            for index, item in enumerate(
                practice_items,
                start=1
            ):
                prompt = item.get(
                    "prompt",
                    ""
                )

                answer = item.get(
                    "answer",
                    ""
                )

                st.markdown(
                    f"**{index}. {prompt}**"
                )

                with st.expander(
                    "Show answer"
                ):
                    st.markdown(
                        str(answer)
                    )


        # Store a readable version in chat history.
        practice_history_parts = [
            "### "
            + practice_type.replace(
                "_",
                " "
            ).title()
        ]

        for index, item in enumerate(
            practice_items,
            start=1
        ):
            practice_history_parts.append(
                f"\n\n**{index}. "
                f"{item.get('prompt', '')}**"
                f"\n\nAnswer: "
                f"{item.get('answer', '')}"
            )

        st.session_state.tutor_messages.append(
            {
                "role": "assistant",
                "content": "".join(
                    practice_history_parts
                ),
            }
        )

        return

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
