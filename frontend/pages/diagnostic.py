"""Handles learning path selection and the initial diagnostic flow."""

import streamlit as st

from api_client import start_diagnostic


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
    Display a temporary placeholder for the next
    diagnostic assessment stage.
    """

    selected_path = st.session_state.get(
        "selected_path"
    )

    path = LEARNING_PATHS.get(
        selected_path,
        {},
    )

    st.markdown(
        '<div class="mc-eyebrow">Initial Diagnostic</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mc-page-title">
            Let's check your current knowledge
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mc-page-description">
            You're starting the
            <strong>{path.get("name", "selected")}</strong>
            learning path.
            Complete this short assessment so MindCompass
            can determine the best starting point for you.
        </div>
        """,
        unsafe_allow_html=True,
    )

    questions = st.session_state.get(
        "diagnostic_questions",
        [],
    )

    st.info(
        f"{len(questions)} diagnostic questions are ready."
    )

    st.caption(
        "The assessment questions will appear here next."
    )

    # Allow the learner to return to path selection
    # before submitting any diagnostic answers.
    if st.button(
        "← Change Learning Path",
        use_container_width=False,
    ):
        st.session_state["diagnostic_started"] = False
        st.session_state["diagnostic_questions"] = []
        st.session_state["diagnostic_topics"] = []

        st.rerun()