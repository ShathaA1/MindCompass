"""Displays the personalized path, progress, and recommended next step."""

import streamlit as st

from api_client import get_dashboard, get_learning_path


def render():
    st.header("Dashboard")

    token = st.session_state["token"]

    try:
        response = get_dashboard(token)

        if response.status_code != 200:
            st.error("Unable to load dashboard.")
            return

        data = response.json()

        st.subheader("Learning Profile")

        st.write("Goal:", data.get("goal", "-"))
        st.write("Current Level:", data.get("current_level", "-"))
        st.write("Weekly Hours:", data.get("weekly_hours", "-"))
        st.write("Topics Assessed:", data.get("topics_assessed", 0))

        st.markdown("---")

        st.subheader("Learning Path")

        path_response = get_learning_path(token)

        if path_response.status_code == 200:
            path_data = path_response.json()

            if path_data.get("learning_path") is None:
                st.info(
                    "Your personalized learning path has not been generated yet."
                )
            else:
                learning_path = path_data["learning_path"]

                st.write(
                    "Path:",
                    learning_path.get("name", "Personalized Learning Path"),
                )

                items = path_data.get("items", [])

                for item in items:
                    st.write(
                        f"{item.get('position')}. "
                        f"{item.get('topic_name')}"
                    )

    except Exception as error:
        st.error(f"Connection error: {error}")