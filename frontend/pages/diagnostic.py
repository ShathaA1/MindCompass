"""Handles learning path selection and the initial diagnostic flow."""

import streamlit as st

from api_client import start_diagnostic, submit_diagnostic


LEARNING_PATHS = {
    "python": {
        "name": "Python",
        "icon": "🐍",
        "description": (
            "Build a strong foundation in Python "
            "programming and problem solving."
        ),
    },
    "machine_learning": {
        "name": "Machine Learning",
        "icon": "🧠",
        "description": (
            "Learn the core concepts needed to "
            "understand machine learning."
        ),
    },
    "agentic_ai": {
        "name": "Agentic AI",
        "icon": "🤖",
        "description": (
            "Learn how AI agents reason, plan, "
            "use tools, and take action."
        ),
    },
}


def render():
    """
    Render learning path selection or the diagnostic stage.
    """

    # Check authentication before showing the setup flow.
    token = st.session_state.get("token")

    if not token:
        st.error(
            "Please log in before starting your learning setup."
        )
        return

    # If the diagnostic has already started,
    # move to the diagnostic assessment stage.
    if st.session_state.get("diagnostic_started"):
        render_diagnostic_stage()
        return

    # Otherwise, show learning path selection.
    render_path_selection(token)


def render_path_selection(token):
    """
    Display the available MindCompass learning paths.
    """

    # ---------- Page introduction ----------

    st.markdown(
        '<div class="mc-eyebrow">Learning Setup</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mc-page-title">
            Choose your learning path
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mc-page-description">
            Select the area you want to learn.
            MindCompass will assess your current knowledge
            and personalize the path based on your results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Learning path cards ----------

    python_col, ml_col, agent_col = st.columns(3)

    path_columns = {
        "python": python_col,
        "machine_learning": ml_col,
        "agentic_ai": agent_col,
    }

    selected_path = st.session_state.get("selected_path")

    for path_key, path in LEARNING_PATHS.items():
        with path_columns[path_key]:

            # Build the card HTML without leading indentation.
            card_html = (
                '<div class="mc-path-card">'
                f'<div class="mc-path-icon">{path["icon"]}</div>'
                f'<div class="mc-path-title">{path["name"]}</div>'
                f'<div class="mc-path-description">{path["description"]}</div>'
                '</div>'
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True,
            )

            # Check whether this learning path is currently selected.
            is_selected = selected_path == path_key

            # Selected paths use the primary button style.
            button_label = (
                f"✓ {path['name']} Selected"
                if is_selected
                else f"Choose {path['name']}"
            )

            if st.button(
                button_label,
                key=f"choose_{path_key}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state["selected_path"] = path_key
                st.rerun()

    # ---------- Continue to diagnostic ----------

    selected_path = st.session_state.get("selected_path")

    if selected_path:
        st.write("")

        if st.button(
            "Continue to Diagnostic →",
            type="primary",
            use_container_width=True,
        ):
            start_selected_diagnostic(
                token=token,
                selected_path=selected_path,
            )

    else:
        st.caption(
            "Choose a learning path to continue."
    )

def start_selected_diagnostic(
    token,
    selected_path,
):
    """
    Request the initial diagnostic from the Tutor Agent.
    """

    with st.spinner(
        "Preparing your diagnostic assessment..."
    ):
        try:
            response = start_diagnostic(
                token=token,
                selected_path=selected_path,
            )

        except Exception as exc:
            st.error(
                "Could not connect to the MindCompass API."
            )
            st.exception(exc)
            return

    # Handle API errors before using response data.
    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                "Unable to start diagnostic.",
            )
        except ValueError:
            detail = "Unable to start diagnostic."

        st.error(detail)
        return
    


    data = response.json()

    # Store the diagnostic data so it remains
    # available across Streamlit reruns.
    st.session_state["diagnostic_topics"] = data.get(
        "diagnostic_topics",
        [],
    )

    st.session_state["diagnostic_questions"] = data.get(
        "assessment_questions",
        [],
    )

    st.session_state["diagnostic_started"] = True

    st.rerun()


def render_diagnostic_stage():
    """
    Render diagnostic questions one at a time and
    temporarily store the learner's answers in session state.
    """

    # Get the selected learning path.
    selected_path = st.session_state.get("selected_path")

    path = LEARNING_PATHS.get(
        selected_path,
        {},
    )

    # Get diagnostic questions prepared by the backend.
    questions = st.session_state.get(
        "diagnostic_questions",
        [],
    )

    # Stop if no questions were returned.
    if not questions:
        st.warning(
            "No diagnostic questions are available for this path."
        )

        if st.button("← Change Learning Path"):
            reset_diagnostic()

        return


    # ---------- Back to learning path selection ----------

    if st.button(
        "← Back to learning paths",
        key="back_to_learning_paths",
    ):
        reset_diagnostic()

    st.write("")

    # Initialize the current question index.
    if "diagnostic_question_index" not in st.session_state:
        st.session_state["diagnostic_question_index"] = 0

    # Initialize temporary learner answers.
    if "diagnostic_answers" not in st.session_state:
        st.session_state["diagnostic_answers"] = {}

    current_index = st.session_state[
        "diagnostic_question_index"
    ]

    total_questions = len(questions)

    current_question = questions[current_index]

    # ---------- Page heading ----------

    st.markdown(
        '<div class="mc-eyebrow">Initial Diagnostic</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mc-page-title">
            {path.get("name", "Learning")} Diagnostic
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mc-page-description">
            Answer each question based on your current knowledge.
            Your results will help MindCompass personalize
            your learning path.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Progress ----------

    progress = (current_index + 1) / total_questions

    st.progress(progress)

    st.caption(
        f"Question {current_index + 1} of {total_questions}"
    )

    # ---------- Question ----------

    question_text = current_question.get(
        "question_text",
        "Question unavailable.",
    )

    # Build the question card without leading indentation.
    # This prevents Streamlit from rendering the HTML as code.
    question_card_html = (
        '<div class="mc-question-card">'
        f'<div class="mc-question-number">'
        f'Question {current_index + 1}'
        '</div>'
        f'<div class="mc-question-text">'
        f'{question_text}'
        '</div>'
        '</div>'
    )

    st.markdown(
        question_card_html,
        unsafe_allow_html=True,
    )

    # ---------- Answer options ----------

    options = current_question.get("options") or []

    if not options:
        st.warning(
            "No answer options are available for this question."
        )
        return

    # Get the learner's previously selected answer.
    saved_answer = st.session_state[
        "diagnostic_answers"
    ].get(current_index)

    # Display answer choices in a 2x2 grid.
    for row_start in range(0, len(options), 2):
        left_col, right_col = st.columns(2)

        row_options = options[row_start:row_start + 2]

        for option_index, (column, option) in enumerate(
            zip(
                [left_col, right_col],
                row_options,
            )
        ):
            with column:
                is_selected = saved_answer == option

                # Highlight the currently selected answer.
                button_label = (
                    f"✓ {option}"
                    if is_selected
                    else option
                )

                if st.button(
                    button_label,
                    key=(
                        f"answer_"
                        f"{current_index}_"
                        f"{row_start + option_index}"
                    ),
                    type=(
                        "primary"
                        if is_selected
                        else "secondary"
                    ),
                    use_container_width=True,
                ):
                    # Save the selected answer for this question.
                    st.session_state[
                        "diagnostic_answers"
                    ][current_index] = option

                    st.rerun()

    # Retrieve the selected answer for navigation validation.
    selected_answer = st.session_state[
        "diagnostic_answers"
    ].get(current_index)

    st.write("")

    # ---------- Navigation ----------

    previous_col, spacer_col, next_col = st.columns(
        [1, 2, 1]
    )

    with previous_col:
        if current_index > 0:
            if st.button(
                "← Previous",
                use_container_width=True,
            ):
                st.session_state[
                    "diagnostic_question_index"
                ] -= 1

                st.rerun()

    with next_col:

        # Show Next until the final question.
        if current_index < total_questions - 1:

            if st.button(
                "Next →",
                type="primary",
                use_container_width=True,
                disabled=selected_answer is None,
            ):
                st.session_state[
                    "diagnostic_question_index"
                ] += 1

                st.rerun()

        # Show Submit on the final question.
        else:
            if st.button(
                "Submit Diagnostic",
                type="primary",
                use_container_width=True,
                disabled=selected_answer is None,
            ):
                submit_diagnostic_answers(
                    token=st.session_state["token"],
                    selected_path=selected_path,
                    questions=questions,
                )


def submit_diagnostic_answers(
    token,
    selected_path,
    questions,
):
    """
    Submit the learner's diagnostic answers to the backend
    and store the personalized learning path result.
    """

    saved_answers = st.session_state.get(
        "diagnostic_answers",
        {},
    )

    # Make sure every diagnostic question has been answered.
    if len(saved_answers) != len(questions):
        st.warning(
            "Please answer all diagnostic questions before submitting."
        )
        return

    # Convert frontend answers into the format expected
    # by the diagnostic submission API.
    assessment_answers = []

    for question_index in range(len(questions)):
        assessment_answers.append(
            {
                "learner_answer": saved_answers[
                    question_index
                ]
            }
        )

    with st.spinner(
        "Analyzing your results and personalizing your learning path..."
    ):
        try:
            response = submit_diagnostic(
                token=token,
                selected_path=selected_path,
                assessment_questions=questions,
                assessment_answers=assessment_answers,
            )

        except Exception as exc:
            st.error(
                "Could not connect to the MindCompass API."
            )
            st.exception(exc)
            return

    # Handle backend validation or processing errors.
    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                "Unable to submit diagnostic.",
            )
        except ValueError:
            detail = "Unable to submit diagnostic."

        st.error(detail)
        return

    data = response.json()

    # Store the backend result temporarily so the next
    # page can use the newly created learning path.
    st.session_state[
        "diagnostic_result"
    ] = data

    # Mark the learner setup as complete.
    st.session_state["setup_complete"] = True

    # Clear temporary diagnostic state.
    for key in [
        "diagnostic_topics",
        "diagnostic_questions",
        "diagnostic_started",
        "diagnostic_question_index",
        "diagnostic_answers",
        "diagnostic_ready_to_submit",
    ]:
        st.session_state.pop(key, None)

    # Move the learner to the Learning Path page.
    st.session_state["page"] = "Learning Path"

    st.rerun()


def reset_diagnostic():
    """
    Clear diagnostic state and return to learning path selection.
    """

    keys_to_clear = [
        "selected_path",
        "diagnostic_topics",
        "diagnostic_questions",
        "diagnostic_started",
        "diagnostic_question_index",
        "diagnostic_answers",
        "diagnostic_ready_to_submit",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.rerun()