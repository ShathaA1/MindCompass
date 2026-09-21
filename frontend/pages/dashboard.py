"""Displays the personalized path, progress, and recommended next step."""

import streamlit as st

from api_client import get_dashboard, get_learning_path


def render():
    """
    Display the learner dashboard.
    """

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
        """
        <div class="mc-page-description">
            Track your learning profile, current path,
            and assessment progress.
        </div>
        """,
        unsafe_allow_html=True,
    )

    token = st.session_state["token"]

    try:
        response = get_dashboard(token)

        if response.status_code != 200:
            st.error("Unable to load dashboard.")
            return

        data = response.json()

        # ---------- Learning profile ----------

        st.subheader("Learning Profile")

        level_col, hours_col, assessed_col = st.columns(3)

        with level_col:
            st.metric(
                "Current Level",
                str(data.get("current_level", "-")).title(),
            )

        with hours_col:
            st.metric(
                "Weekly Hours",
                data.get("weekly_hours", "-"),
            )

        with assessed_col:
            st.metric(
                "Topics Assessed",
                data.get("topics_assessed", 0),
            )

        st.markdown(
            f"""
            <div class="mc-card">
                <div class="mc-card-title">
                    Learning Goal
                </div>
                <div class="mc-card-text">
                    {data.get("goal", "-")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- Personalized learning path ----------

        st.subheader("Learning Path")

        path_response = get_learning_path(token)

        if path_response.status_code != 200:
            st.error(
                "Unable to load learning path."
            )
            return

        path_data = path_response.json()

        if path_data.get("learning_path") is None:
            st.info(
                "Your personalized learning path "
                "has not been generated yet."
            )
            return

        learning_path = path_data["learning_path"]

        st.markdown(
            f"""
            <div class="mc-card">
                <div class="mc-card-title">
                    {learning_path.get(
                        "name",
                        "Personalized Learning Path",
                    )}
                </div>
                <div class="mc-card-text">
                    Your current personalized learning path.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        items = path_data.get("items", [])

        for item in items:
            position = item.get("position", "")
            topic_name = item.get(
                "topic_name",
                "Topic",
            )

            st.markdown(
                f"""
                <div class="mc-card">
                    <div class="mc-card-title">
                        {position}. {topic_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as error:
        st.error(
            f"Connection error: {error}"
        )