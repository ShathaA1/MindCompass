"""Configures the main Streamlit application and learner navigation."""

import streamlit as st

st.set_page_config(
    page_title="MindCompass",
    page_icon="🧭",
    layout="wide",
)

st.title("MindCompass")
st.subheader("Your Personalized Agentic AI Tutor")

st.write(
    "MindCompass creates and continuously adapts a personalized "
    "learning path based on your goals, level, and progress."
)
