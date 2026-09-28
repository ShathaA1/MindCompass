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

    recommended_action = result.get(
        "recommended_action"
    )

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

    # When the learner has mastered the current topic,
    # allow them to move to the next topic explicitly.
    if recommended_action == "recommend":
        st.info(
            "You have mastered this topic and are ready "
            "to continue to the next topic."
        )

        if st.button(
            "Continue to Next Topic",
            type="primary",
            key="assessment_continue_next_topic",
        ):
            st.session_state.pop(
                "chat_session_id",
                None
            )

            st.session_state.pop(
                "chat_session_topic_id",
                None
            )

            st.session_state.pop(
                "tutor_messages",
                None
            )

            st.session_state.pop(
                "latest_recommended_action",
                None
            )

            st.session_state.pop(
                "pending_next_topic_id",
                None
            )

            st.rerun()

        return

    # For explanation, practice, review, or reassessment,
    # return to the normal Tutor conversation so the
    # learner can continue typing.
    st.rerun()




def render_stored_message(message: dict) -> None:
    """
    Render a stored Tutor message according to
    the agent action that originally produced it.
    """

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )

    agent_action = message.get(
        "agent_action"
    )

    with st.chat_message(role):

        # Render stored practice using the same
        # structured UI used for new practice.
        if (
            role == "assistant"
            and agent_action == "practice"
        ):
            try:
                practice_data = json.loads(
                    content
                )
            except json.JSONDecodeError:
                st.markdown(content)
                return

            practice_type = practice_data.get(
                "practice_type",
                "practice"
            )

            practice_items = practice_data.get(
                "items",
                []
            )

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

            return

        # Show stored assessments as a simple history
        # message instead of rendering raw quiz JSON.
        if (
            role == "assistant"
            and agent_action == "assess"
        ):
            st.markdown(
                "**Assessment generated for this topic.**"
            )
            return

        # Render normal stored messages as Markdown.
        st.markdown(content)


def render_next_step_message(
    recommended_action: str | None,
) -> None:
    """
    Display a learner-friendly suggestion for
    the next recommended learning step.
    """

    messages = {
        "practice": (
            "If you have any questions about this explanation, "
            "feel free to ask. Otherwise, we can move on to practice."
        ),
        "assess": (
            "If you'd like to clarify anything first, ask me. "
            "Otherwise, we can move on to the assessment."
        ),
        "review": (
            "If anything is still unclear, I can explain it again. "
            "Otherwise, we can review the areas that need more attention."
        ),
        "recommend": (
            "If you have any final questions about this topic, ask me. "
            "Otherwise, we can continue to the next topic."
        ),
    }

    message = messages.get(
        recommended_action
    )

    if message:
        st.caption(message)




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

    # Prefer the latest Tutor recommendation from
    # the current session when it is available.
    recommended_action = (
        st.session_state.get(
            "latest_recommended_action"
        )
        or dashboard_data.get(
            "recommended_action"
        )
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

            # Create a placeholder so the recommendation
            # can be updated after the Tutor responds.
            recommendation_placeholder = st.empty()

            recommendation_placeholder.markdown(
                f"**{recommended_action_text}**"
            )

    # Check whether the learner explicitly opened
    # a previous topic conversation.
    selected_session_id = st.session_state.get(
        "selected_conversation_session_id"
    )

    selected_topic_id = st.session_state.get(
        "selected_conversation_topic_id"
    )


    # ---------------------------------------------------------
    # Chat session initialization
    # ---------------------------------------------------------

    # Load all Tutor chat sessions so they can be used
    # for both topic selection and conversation navigation.
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


    if selected_session_id:
        # Keep the explicitly selected historical
        # conversation open across Streamlit reruns.
        st.session_state.chat_session_id = (
            selected_session_id
        )

        st.session_state.chat_session_topic_id = (
            selected_topic_id
        )

    else:
        if (
            "chat_session_id" not in st.session_state
            or st.session_state.get(
                "chat_session_topic_id"
            ) != current_topic_id
        ):
            # Clear frontend state that belongs
            # to the previously opened topic.
            st.session_state.pop(
                "tutor_messages",
                None
            )

            st.session_state.pop(
                "active_assessment",
                None
            )

            st.session_state.pop(
                "latest_recommended_action",
                None
            )

            # Find an existing session that belongs
            # specifically to the current topic.
            topic_session = next(
                (
                    session
                    for session in sessions
                    if session.get("topic_id")
                    == current_topic_id
                ),
                None,
            )

            if topic_session:
                # Continue the existing conversation
                # for the current topic.
                st.session_state.chat_session_id = (
                    topic_session["session_id"]
                )

            else:
                try:
                    # Create a new conversation for the
                    # current topic when none exists yet.
                    response = create_chat_session(
                        token=token,
                        session_name=current_topic_name,
                        learning_path_id=dashboard_data[
                            "learning_path"
                        ]["learning_path_id"],
                        topic_id=current_topic_id,
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

                # Add the new session to the local list
                # so it is available during this render.
                sessions.insert(
                    0,
                    session_data
                )

            # Remember which topic owns the
            # currently opened chat session.
            st.session_state[
                "chat_session_topic_id"
            ] = current_topic_id


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

    for message in st.session_state.tutor_messages:
        render_stored_message(
            message
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


    # Keep free-form chat available so the learner
    # can still ask questions at any time.
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


    next_topic_id = result.get(
        "next_topic_id"
    )

    if next_topic_id:
        st.session_state[
            "pending_next_topic_id"
        ] = next_topic_id


    agent_action = result.get(
        "agent_action"
    )

    # Store the latest recommendation returned
    # by the Tutor Agent for immediate UI updates.
    latest_recommended_action = result.get(
        "recommended_action"
    )

    if latest_recommended_action:
        st.session_state[
            "latest_recommended_action"
        ] = latest_recommended_action

        # Update the recommendation shown in the Tutor
        # header immediately after the Agent responds.
        recommendation_placeholder.markdown(
            f"**{latest_recommended_action.replace('_', ' ').title()}**"
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

        render_next_step_message(
            latest_recommended_action
        )

    # Show a clear transition button when the learner
    # is ready to move to the next topic.
    if st.session_state.get(
        "pending_next_topic_id"
    ):
        if st.button(
            "Continue to Next Topic",
            type="primary",
            key="continue_next_topic",
        ):
            # Clear the current topic session so the
            # next rerun loads the new topic session.
            st.session_state.pop(
                "chat_session_id",
                None
            )

            st.session_state.pop(
                "chat_session_topic_id",
                None
            )

            st.session_state.pop(
                "tutor_messages",
                None
            )

            st.session_state.pop(
                "latest_recommended_action",
                None
            )

            st.session_state.pop(
                "active_assessment",
                None
            )

            st.session_state.pop(
                "pending_next_topic_id",
                None
            )

            st.rerun()


