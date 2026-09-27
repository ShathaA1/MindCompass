"""Displays the learner's personalized learning path."""

import streamlit as st

from api_client import get_learning_path

def get_topic_display_name(
    topic_id: int,
    original_name: str,
) -> str:
    """
    Return a learner-friendly topic name without
    changing the internal database topic name.
    """

    display_names = {
        27: "Python Foundations for Agentic AI",
        26: "Machine Learning Foundations for Agentic AI",
    }

    return display_names.get(
        topic_id,
        original_name,
    )


def get_foundation_roadmap(
    topic_id: int,
) -> list[str]:
    """
    Return a clear roadmap for foundation topics.
    """

    roadmaps = {
        27: [
            "Variables and Data Types",
            "Lists and Dictionaries",
            "Conditions and Loops",
            "Functions",
            "Core Python Behavior",
        ],
        26: [
            "Machine Learning Fundamentals",
            "Features and Labels",
            "Supervised vs Unsupervised Learning",
            "Classification and Regression",
            "Model Training and Evaluation",
        ],
    }

    return roadmaps.get(
        topic_id,
        [],
    )


def format_recommended_action(
    action: str | None,
) -> str:
    """
    Convert internal agent actions into
    learner-friendly next-step labels.
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

        # Separate prerequisite foundation courses
        # from the main Agentic AI curriculum.
        foundation_items = [
            item
            for item in items
            if item.get("topic_id") in [26, 27]
        ]

        core_items = [
            item
            for item in items
            if item.get("topic_id") not in [26, 27]
        ]

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
        # ---------- Foundation courses ----------

        if foundation_items:
            st.markdown(
                '<div class="mc-dashboard-section-title">'
                "Foundation Courses"
                "</div>",
                unsafe_allow_html=True,
            )

            st.caption(
                "These courses were added based on your "
                "diagnostic results and the prerequisites "
                "needed for Agentic AI."
            )

            for item in foundation_items:
                topic_id = item.get("topic_id")

                original_name = item.get(
                    "topic_name",
                    "Foundation Topic",
                )

                topic_name = get_topic_display_name(
                    topic_id,
                    original_name,
                )

                status = item.get(
                    "status",
                    "pending",
                )

                recommended_action = (
                    format_recommended_action(
                        item.get("recommended_action")
                    )
                )

                roadmap = get_foundation_roadmap(
                    topic_id
                )

                if status == "completed":
                    state_label = "Completed"
                    marker = "✓"
                else:
                    state_label = "Current"
                    marker = "→"

                st.markdown(
                    (
                        '<div class="mc-path-item current">'
                        '<div class="mc-path-marker-column">'
                        f'<div class="mc-path-marker">{marker}</div>'
                        "</div>"
                        '<div class="mc-path-content">'
                        '<div class="mc-path-item-header">'
                        f'<div class="mc-path-topic">{topic_name}</div>'
                        f'<div class="mc-path-state">{state_label}</div>'
                        "</div>"
                        '<div class="mc-path-action">'
                        "Recommended next step: "
                        f"<strong>{recommended_action}</strong>"
                        "</div>"
                        "</div>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                if roadmap:
                    with st.expander(
                        f"What you'll cover in {topic_name}"
                    ):
                        for index, topic in enumerate(
                            roadmap,
                            start=1,
                        ):
                            st.write(
                                f"{index}. {topic}"
                            )

        # ---------- Core Agentic AI path ----------

        if core_items:
            st.markdown(
                '<div class="mc-dashboard-section-title">'
                "Core Agentic AI Path"
                "</div>",
                unsafe_allow_html=True,
            )

            st.caption(
                "These are the main Agentic AI topics "
                "you will study after completing any "
                "required foundations."
            )

            current_found = False
            journey_html = (
                '<div class="mc-path-journey">'
            )

            for item in core_items:
                position = item.get(
                    "position",
                    "",
                )

                topic_id = item.get(
                    "topic_id"
                )

                original_name = item.get(
                    "topic_name",
                    "Topic",
                )

                topic_name = get_topic_display_name(
                    topic_id,
                    original_name,
                )

                status = item.get(
                    "status",
                    "pending",
                )

                recommended_action = item.get(
                    "recommended_action"
                )

                # Determine how the topic should appear.
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

                if (
                    recommended_action
                    and status != "completed"
                ):
                    action_label = (
                        format_recommended_action(
                            recommended_action
                        )
                    )

                    action_html = (
                        '<div class="mc-path-action">'
                        "Recommended: "
                        f"<strong>{action_label}</strong>"
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