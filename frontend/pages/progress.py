"""Displays topic mastery, completed topics, and identified weak areas."""

import streamlit as st

from api_client import get_progress


def render():
    st.header("Progress")

    try:
        response = get_progress(st.session_state["token"])

        if response.status_code != 200:
            st.error("Unable to load progress.")
            return

        topics = response.json()

        if not topics:
            st.info(
                "No progress data yet. "
                "Complete assessments to start tracking your progress."
            )
            return

        for topic in topics:
            st.subheader(topic.get("topic_name", "Topic"))

            mastery = topic.get("mastery_score", 0)

            st.write(f"Mastery Score: {mastery}%")

            weak_areas = topic.get("weak_areas")

            if weak_areas:
                st.write("Weak Areas:", weak_areas)

            st.markdown("---")

    except Exception as error:
        st.error(f"Connection error: {error}")