"""Collects the learner's goal, level, availability, and preferences."""

import streamlit as st

from api_client import create_onboarding


LEARNING_STYLES = [
    "Concise Explanations",
    "Detailed Explanations",
    "Practical Examples",
    "Explanations with Examples",
]


def render():
    """Display the learner onboarding form."""

    st.header("Onboarding")

    st.write(
        "Tell MindCompass how you prefer to learn."
    )

    goal = st.text_input(
        "Learning Goal"
    )

    level = st.selectbox(
        "Current Level",
        [
            "beginner",
            "intermediate",
            "advanced",
        ],
    )

    weekly_hours = st.number_input(
        "Weekly Learning Hours",
        min_value=1,
        max_value=40,
        value=5,
    )

    learning_style = st.selectbox(
        "Preferred Learning Style",
        LEARNING_STYLES,
    )

    pace = st.selectbox(
        "Preferred Pace",
        [
            "slow",
            "moderate",
            "fast",
        ],
        index=1,
    )

    if st.button("Save Profile"):
        if not goal.strip():
            st.warning(
                "Please enter your learning goal."
            )
            return

        data = {
            "goal": goal.strip(),
            "initial_level": level,
            "weekly_hours": int(weekly_hours),
            "preferred_format": learning_style,
            "preferred_pace": pace,
        }

        try:
            response = create_onboarding(
                st.session_state["token"],
                data,
            )

            if response.status_code == 200:
                st.success(
                    "Profile created successfully."
                )

                st.session_state["profile_complete"] = True
                st.session_state["page"] = "Dashboard"

                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()

            else:
                detail = response.json().get(
                    "detail",
                    "Unable to save profile.",
                )

                st.error(detail)

        except Exception as error:
            st.error(
                f"Connection error: {error}"
            )