"""Displays the learner's personalized learning path."""

import streamlit as st

from api_client import (
    get_chat_sessions,
    get_learning_path,
)


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

        # Load existing Tutor conversations so completed
        # topics can be reopened from the learning path.
        sessions_response = get_chat_sessions(
            token
        )

        if sessions_response.status_code == 200:
            chat_sessions = sessions_response.json().get(
                "sessions",
                [],
            )
        else:
            chat_sessions = []


        # Map each topic to its most recent Tutor session.
        session_by_topic_id = {}

        for session in chat_sessions:
            session_topic_id = session.get(
                "topic_id"
            )

            if (
                session_topic_id is not None
                and session_topic_id not in session_by_topic_id
            ):
                session_by_topic_id[
                    session_topic_id
                ] = session

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
                    visual_state = "completed"
                    state_label = ""
                    marker = "✓"
                else:
                    visual_state = "current"
                    state_label = "Current"
                    marker = "→"

                # Build the optional state label before
                # rendering the topic card.
                state_html = ""

                if state_label:
                    state_html = (
                        f'<div class="mc-path-state">{state_label}</div>'
                    )

                st.markdown(
                    (
                        f'<div class="mc-path-item {visual_state}">'
                        '<div class="mc-path-marker-column">'
                        f'<div class="mc-path-marker">{marker}</div>'
                        "</div>"
                        '<div class="mc-path-content">'
                        '<div class="mc-path-item-header">'
                        f'<div class="mc-path-topic">{topic_name}</div>'
                        f"{state_html}"
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
                    state_label = ""
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

                # Build the optional state label before
                # rendering the topic card.
                state_html = ""

                if state_label:
                    state_html = (
                        f'<div class="mc-path-state">{state_label}</div>'
                    )

                # Find an existing Tutor conversation
                # for this topic, if one exists.
                topic_session = session_by_topic_id.get(
                    topic_id
                )

                # Keep the topic card and conversation
                # action aligned on the same row.
                topic_col, button_col = st.columns(
                    [5.5, 1.5],
                    vertical_alignment="center",
                )

                with topic_col:
                    st.markdown(
                        (
                            f'<div class="mc-path-item {visual_state}">'
                            '<div class="mc-path-marker-column">'
                            f'<div class="mc-path-marker">{marker}</div>'
                            '<div class="mc-path-line"></div>'
                            "</div>"
                            '<div class="mc-path-content">'
                            '<div class="mc-path-item-header">'
                            f'<div class="mc-path-topic">{topic_name}</div>'
                            f"{state_html}"
                            "</div>"
                            f"{action_html}"
                            "</div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                with button_col:
                    # Show a conversation action only when
                    # this topic already has a Tutor session.
                    if topic_session:
                        button_label = (
                            "Resume Conversation"
                            if visual_state == "current"
                            else "View Conversation"
                        )

                        if st.button(
                            button_label,
                            key=f"topic_chat_{topic_id}",
                            width="stretch",
                        ):
                            # Store the selected conversation so
                            # the Tutor page opens the correct session.
                            st.session_state[
                                "selected_conversation_session_id"
                            ] = topic_session["session_id"]

                            st.session_state[
                                "selected_conversation_topic_id"
                            ] = topic_id

                            st.session_state[
                                "selected_conversation_name"
                            ] = topic_name

                            # Clear previous Tutor UI state so
                            # the selected conversation can reload.
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

                            # Navigate directly to the Tutor page.
                            st.session_state[
                                "page"
                            ] = "Tutor"

                            st.rerun()


    except Exception as error:
        st.error(
            f"Connection error: {error}"
        )