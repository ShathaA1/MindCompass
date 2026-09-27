"""Collects the learner's goal, level, availability, and preferences."""

import streamlit as st

from api_client import create_onboarding


LEARNING_STYLES = [
    "Concise Explanations",
    "Detailed Explanations",
    "Practical Examples",
    "Guided Practice",
]


def render():
    """
    Display the learner onboarding form.
    """

    # ---------- Page introduction ----------

    st.markdown(
        '<div class="mc-eyebrow">Getting Started</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mc-page-title">'
        "Personalize your learning experience"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mc-page-description">
            Tell MindCompass about your learning preferences.
            We'll use them to adapt your learning experience.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mc-card">
            <div class="mc-card-title">
                Your learning profile
            </div>
            <div class="mc-card-text">
                These preferences help the tutor understand
                how you want to learn.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Learning goal ----------

    goal = st.text_input(
        "Learning Goal",
        placeholder=(
            "Example: Build practical AI engineering skills"
        ),
    )

    # ---------- Level and weekly availability ----------

    level_col, hours_col = st.columns(2)

    with level_col:
        level = st.selectbox(
            "Current Level",
            [
                "beginner",
                "intermediate",
                "advanced",
            ],
        )

    with hours_col:
        weekly_hours = st.number_input(
            "Weekly Learning Hours",
            min_value=1,
            max_value=40,
            value=5,
        )

    # ---------- Learning preferences ----------

    style_col, pace_col = st.columns(2)

    with style_col:
        learning_style = st.selectbox(
            "Preferred Learning Style",
            LEARNING_STYLES,
        )

    with pace_col:
        pace = st.selectbox(
            "Preferred Pace",
            [
                "slow",
                "moderate",
                "fast",
            ],
            index=1,
        )

    st.write("")

    # ---------- Save learner profile ----------

    if st.button(
        "Continue →",
        type="primary",
        width="stretch",
    ):
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

                # Continue directly to learning path selection
                # and the initial diagnostic.
                st.session_state["page"] = "Diagnostic"

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
