"""Displays the learner's personalized learning path."""

import streamlit as st

from api_client import get_learning_path


def render():
    """Display the learner's personalized learning path."""

    # ---------- Page introduction ----------

    st.markdown(
        '<div class="mc-eyebrow">Learning Path</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mc-page-title">'
        "Your personalized learning path"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="mc-page-description">'
            "Follow your topics in order and see what "
            "you have completed and what comes next."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    token = st.session_state["token"]

    try:
        # Load the learner's personalized learning path.
        response = get_learning_path(token)

        if response.status_code != 200:
            st.error("Unable to load learning path.")
            return

        data = response.json()
        learning_path = data.get("learning_path")

        if learning_path is None:
            st.info(
                "Your personalized learning path "
                "has not been generated yet."
            )
            return

        items = data.get("items", [])

        if not items:
            st.info(
                "No topics are available in this learning path yet."
            )
            return

        # ---------- Prepare learning path data ----------

        path_name = learning_path.get(
            "name",
            "Personalized Learning Path",
        )

        path_status = learning_path.get(
            "status",
            "active",
        )

        total_topics = len(items)

        completed_topics = sum(
            1
            for item in items
            if item.get("status") == "completed"
        )

        progress = round(
            (completed_topics / total_topics) * 100,
            1,
        )

        # ---------- Learning path summary ----------

        st.markdown(
            (
                '<div class="mc-path-summary">'
                '<div class="mc-path-summary-header">'
                '<div>'
                '<div class="mc-dashboard-label">'
                "CURRENT LEARNING PATH"
                "</div>"
                '<div class="mc-path-name">'
                f"{path_name}"
                "</div>"
                "</div>"
                '<div class="mc-dashboard-status">'
                f"{path_status.upper()}"
                "</div>"
                "</div>"
                '<div class="mc-dashboard-progress-header">'
                "<span>Overall Progress</span>"
                f"<span>{progress}%</span>"
                "</div>"
                '<div class="mc-dashboard-progress-track">'
                '<div class="mc-dashboard-progress-fill" '
                f'style="width: {progress}%;"></div>'
                "</div>"
                '<div class="mc-dashboard-progress-caption">'
                f"{completed_topics} of {total_topics} "
                "topics completed"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        # ---------- Learning journey ----------

        st.markdown(
            '<div class="mc-dashboard-section-title">'
            "Learning Journey"
            "</div>",
            unsafe_allow_html=True,
        )

        current_found = False
        journey_html = '<div class="mc-path-journey">'

        for item in items:
            position = item.get(
                "position",
                "",
            )

            topic_name = item.get(
                "topic_name",
                "Topic",
            )

            status = item.get(
                "status",
                "pending",
            )

            recommended_action = item.get(
                "recommended_action"
            )

            # Determine how the topic should appear in the journey.
            if status == "completed":
                visual_state = "completed"
                state_label = "Completed"
                marker = "✓"

            elif not current_found:
                visual_state = "current"
                state_label = "Current"
                marker = str(position)
                current_found = True

            else:
                visual_state = "upcoming"
                state_label = "Upcoming"
                marker = str(position)

            action_html = ""

            # Show the recommendation only for topics
            # that still require learner action.
            if recommended_action and status != "completed":
                action_html = (
                    '<div class="mc-path-action">'
                    "Recommended: "
                    f"<strong>{recommended_action.title()}</strong>"
                    "</div>"
                )

            journey_html += (
                f'<div class="mc-path-item {visual_state}">'
                '<div class="mc-path-marker-column">'
                f'<div class="mc-path-marker">{marker}</div>'
                '<div class="mc-path-line"></div>'
                "</div>"
                '<div class="mc-path-content">'
                '<div class="mc-path-item-header">'
                f'<div class="mc-path-topic">{topic_name}</div>'
                f'<div class="mc-path-state">{state_label}</div>'
                "</div>"
                f"{action_html}"
                "</div>"
                "</div>"
            )

        journey_html += "</div>"

        st.markdown(
            journey_html,
            unsafe_allow_html=True,
        )

    except Exception as error:
        st.error(
            f"Connection error: {error}"
        )