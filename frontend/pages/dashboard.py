"""Displays the learner's personalized progress and next learning step."""

import streamlit as st

from api_client import get_dashboard



def get_topic_display_name(
    topic_id: int | None,
    original_name: str,
) -> str:
    """
    Return a learner-friendly topic name without
    changing the internal database topic name.
    """

    display_names = {
        1: "Python Diagnostic",
        2: "Machine Learning Diagnostic",
        27: "Python Foundations for Agentic AI",
        26: "Machine Learning Foundations for Agentic AI",
    }

    return display_names.get(
        topic_id,
        original_name,
    )


def format_recommended_action(
    action: str | None,
) -> str:
    """
    Convert the internal agent action into
    a learner-friendly recommendation.
    """

    action_labels = {
        "explain": "Learn the concept",
        "practice": "Practice",
        "review": "Review weak areas",
        "assess": "Take an assessment",
        "recommend": "Move to the next topic",
    }

    if not action:
        return "Continue learning"

    return action_labels.get(
        action,
        action.replace("_", " ").title(),
    )


def format_weak_area(
    weak_area,
) -> tuple[str, str]:
    """
    Convert a stored weak-area value into a readable
    area name and optional explanation.
    """

    if isinstance(weak_area, dict):
        area = weak_area.get(
            "area",
            "Learning gap",
        )

        reason = weak_area.get(
            "reason",
            "",
        )

        return area, reason

    return str(weak_area), ""

def render():
    """Display the learner dashboard."""

    # ---------- Page introduction ----------

    st.markdown(
        '<div class="mc-eyebrow">Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mc-page-title">'
        "Your learning journey"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="mc-page-description">'
            "Continue where you left off and track your progress."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    token = st.session_state["token"]

    try:
        # Load the personalized dashboard summary.
        response = get_dashboard(token)

        if response.status_code != 200:
            st.error("Unable to load dashboard.")
            return

        data = response.json()

        learning_path = data.get("learning_path")

        if learning_path is None:
            st.info(
                "Your personalized learning path "
                "has not been generated yet."
            )
            return

        # ---------- Prepare dashboard data ----------

        path_name = learning_path.get(
            "name",
            "Personalized Learning Path",
        )

        path_status = learning_path.get(
            "status",
            "active",
        )

        progress = data.get(
            "progress_percentage",
            0,
        )

        completed_topics = data.get(
            "completed_topics",
            0,
        )

        total_topics = data.get(
            "total_topics",
            0,
        )


        current_topic = data.get("current_topic")
        recommended_action = data.get("recommended_action")

        topic_masteries = data.get(
            "topic_masteries",
            [],
        )

        # Exclude diagnostic-only mastery records
        # from learner-facing progress calculations.
        visible_masteries = [
            mastery
            for mastery in topic_masteries
            if mastery.get("topic_id") not in [1, 2]
        ]

        # Calculate average mastery using only actual
        # learning topics shown to the learner.
        average_mastery = (
            round(
                sum(
                    mastery.get(
                        "mastery_score",
                        0,
                    )
                    for mastery in visible_masteries
                ) / len(visible_masteries),
                1,
            )
            if visible_masteries
            else None
        )

        # Count only learner-facing assessed topics.
        topics_assessed = len(
            visible_masteries
        )

        # Show weak areas only for the learner's
        # current learning topic.
        current_topic_id = (
            current_topic.get("topic_id")
            if current_topic
            else None
        )

        current_topic_mastery = next(
            (
                mastery
                for mastery in topic_masteries
                if mastery.get("topic_id")
                == current_topic_id
            ),
            None,
        )

        weak_areas = (
            current_topic_mastery.get(
                "weak_areas",
                [],
            )
            if current_topic_mastery
            else []
        )

        current_topic_name = (
            get_topic_display_name(
                current_topic.get("topic_id"),
                current_topic.get(
                    "topic_name",
                    "Current Topic",
                ),
            )
            if current_topic
            else "Learning Path Completed"
        )

        action_label = format_recommended_action(
        recommended_action
        )

        mastery_label = (
            f"{average_mastery}%"
            if average_mastery is not None
            else "-"
        )

        # ---------- Current learning path ----------

        st.markdown(
            (
                '<div class="mc-dashboard-path">'
                '<div class="mc-dashboard-path-header">'
                '<div>'
                '<div class="mc-dashboard-label">'
                "CURRENT LEARNING PATH"
                "</div>"
                '<div class="mc-dashboard-path-title">'
                f"{path_name}"
                "</div>"
                "</div>"
                '<div class="mc-dashboard-status">'
                f"{path_status.upper()}"
                "</div>"
                "</div>"
                '<div class="mc-dashboard-progress-header">'
                "<span>Progress</span>"
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

        # ---------- Continue learning and progress ----------

        left_col, right_col = st.columns(
            [1.4, 1],
            gap="large",
        )

        with left_col:
            st.markdown(
                (
                    '<div class="mc-dashboard-panel">'
                    '<div class="mc-dashboard-label">'
                    "CONTINUE LEARNING"
                    "</div>"
                    '<div class="mc-dashboard-topic-title">'
                    f"{current_topic_name}"
                    "</div>"
                    '<div class="mc-dashboard-muted">'
                    "Recommended next action"
                    "</div>"
                    '<div class="mc-dashboard-action">'
                    f"{action_label}"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            if current_topic:
                if st.button(
                    "Continue Learning",
                    type="primary",
                    width="stretch",
                    key="dashboard_continue",
                ):
                    st.session_state["page"] = "Tutor"
                    st.rerun()

        with right_col:
            st.markdown(
                (
                    '<div class="mc-dashboard-panel">'
                    '<div class="mc-dashboard-label">'
                    "YOUR PROGRESS"
                    "</div>"
                    '<div class="mc-dashboard-stat-row">'
                    "<span>Average Mastery</span>"
                    f"<strong>{mastery_label}</strong>"
                    "</div>"
                    '<div class="mc-dashboard-stat-row">'
                    "<span>Topics Completed</span>"
                    f"<strong>{completed_topics}/{total_topics}</strong>"
                    "</div>"
                    '<div class="mc-dashboard-stat-row">'
                    "<span>Topics Assessed</span>"
                    f"<strong>{topics_assessed}</strong>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        # ---------- Topic mastery ----------

        st.markdown(
            '<div class="mc-dashboard-section-title">'
            "Topic Mastery"
            "</div>",
            unsafe_allow_html=True,
        )


        visible_masteries = [
            mastery
            for mastery in topic_masteries
            if mastery.get("topic_id") not in [1, 2]
        ]

        if visible_masteries:
            mastery_html = (
                '<div class="mc-dashboard-panel '
                'mc-dashboard-mastery-panel">'
            )

            for mastery in visible_masteries:
                topic_id = mastery.get(
                "topic_id"
                )

                topic_name = get_topic_display_name(
                    topic_id,
                    mastery.get(
                        "topic_name",
                        "Topic",
                    ),
                )

                score = mastery.get(
                    "mastery_score",
                    0,
                )

                
                mastery_html += (
                    '<div class="mc-dashboard-mastery-item">'
                    '<div class="mc-dashboard-mastery-header">'
                    f"<span>{topic_name}</span>"
                    f"<strong>{score:g}%</strong>"
                    "</div>"
                    '<div class="mc-dashboard-progress-track">'
                    '<div class="mc-dashboard-progress-fill" '
                    f'style="width: {score}%;"></div>'
                    "</div>"
                    "</div>"
                )

            mastery_html += "</div>"

            st.markdown(
                mastery_html,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Complete an assessment to see "
                "your topic mastery."
            )

        # ---------- Areas to improve ----------

        st.markdown(
            '<div class="mc-dashboard-section-title">'
            "Areas to Improve"
            "</div>",
            unsafe_allow_html=True,
        )

        if weak_areas:
            weak_areas_html = ""

            for weak_area in weak_areas:
                area, _ = format_weak_area(
                    weak_area
                )

                weak_areas_html += (
                    '<span class="mc-dashboard-weak-area">'
                    f"{area}"
                    "</span>"
                )

            weak_areas_html += "</div>"

            st.markdown(
                (
                    '<div class="mc-dashboard-panel">'
                    '<div class="mc-dashboard-weak-list">'
                    f"{weak_areas_html}"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                (
                    '<div class="mc-dashboard-panel">'
                    '<div class="mc-dashboard-muted">'
                    "No weak areas identified yet."
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    except Exception as error:
        st.error(
            f"Connection error: {error}"
        )