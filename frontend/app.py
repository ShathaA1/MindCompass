"""Configures the main Streamlit application and learner navigation."""

import streamlit as st

from api_client import get_profile, get_learning_path, login, register
from pages.dashboard import render as render_dashboard
from pages.onboarding import render as render_onboarding
from pages.progress import render as render_progress


st.set_page_config(
    page_title="MindCompass",
    page_icon="🧭",
    layout="wide",
)


def create_columns(spec):
    """Support both old and new Streamlit versions."""
    if hasattr(st, "columns"):
        return st.columns(spec)

    return st.beta_columns(spec)


def rerun_app():
    """Rerun Streamlit while supporting older versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def logout():
    """Clear the current learner session."""
    for key in [
        "token",
        "page",
        "profile_complete",
        "welcome_mode",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    rerun_app()


def show_welcome():
    """Display the welcome screen before authentication."""

    st.image(
        "frontend/assets/mindcompass_logo.png",
        width=500,
    )

    st.markdown(
        """
        <div style="text-align: center;">
            <h1>MindCompass</h1>
            <h3>Your Personalized Agentic AI Tutor</h3>
            <p>
                A personalized learning experience that adapts
                to your goals, level, and progress.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    mode = st.session_state.get("welcome_mode")

    if mode is None:
        login_col, register_col = create_columns(2)

        with login_col:
            if st.button("Login"):
                st.session_state["welcome_mode"] = "Login"
                rerun_app()

        with register_col:
            if st.button("Create Account"):
                st.session_state["welcome_mode"] = "Register"
                rerun_app()

    elif mode == "Login":
        if st.button("← Back"):
            del st.session_state["welcome_mode"]
            rerun_app()

        show_login()

    elif mode == "Register":
        if st.button("← Back"):
            del st.session_state["welcome_mode"]
            rerun_app()

        show_register()


def show_login():
    """Display the login form."""

    st.subheader("Login")

    email = st.text_input(
        "Email",
        key="login_email",
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_password",
    )

    if st.button("Login"):
        if not email or not password:
            st.warning(
                "Please enter your email and password."
            )
            return

        try:
            response = login(
                email,
                password,
            )

            if response.status_code == 200:
                token = response.json()["access_token"]

                st.session_state["token"] = token

                profile_response = get_profile(token)

                if profile_response.status_code == 200:
                    st.session_state["profile_complete"] = True
                    st.session_state["page"] = "Dashboard"

                elif profile_response.status_code == 404:
                    st.session_state["profile_complete"] = False
                    st.session_state["page"] = "Onboarding"

                else:
                    st.error(
                        "Unable to check learner profile."
                    )
                    return

                rerun_app()

            else:
                detail = response.json().get(
                    "detail",
                    "Invalid email or password.",
                )

                st.error(detail)

        except Exception as error:
            st.error(
                f"Connection error: {error}"
            )


def show_register():
    """Display the registration form."""

    st.subheader("Create Account")

    name = st.text_input(
        "Name",
        key="register_name",
    )

    email = st.text_input(
        "Email",
        key="register_email",
    )

    password = st.text_input(
        "Password",
        type="password",
        key="register_password",
    )

    if st.button("Create Account"):
        if not name or not email or not password:
            st.warning(
                "Please complete all fields."
            )
            return

        try:
            response = register(
                name,
                email,
                password,
            )

            if response.status_code == 200:
                st.success(
                    "Account created successfully."
                )

                st.session_state["welcome_mode"] = "Login"

                rerun_app()

            else:
                detail = response.json().get(
                    "detail",
                    "Unable to create account.",
                )

                st.error(detail)

        except Exception as error:
            st.error(
                f"Connection error: {error}"
            )


def show_learning_path():
    """Display the learner's current learning path."""

    st.header("Learning Path")

    try:
        response = get_learning_path(
            st.session_state["token"]
        )

        if response.status_code != 200:
            st.error(
                "Unable to load learning path."
            )
            return

        data = response.json()

        if data.get("learning_path") is None:
            st.info(
                "Your personalized learning path "
                "has not been generated yet."
            )
            return

        learning_path = data["learning_path"]

        st.subheader(
            learning_path.get(
                "name",
                "Personalized Learning Path",
            )
        )

        items = data.get("items", [])

        for item in items:
            position = item.get(
                "position",
                "",
            )

            topic_name = item.get(
                "topic_name",
                "Topic",
            )

            st.write(
                f"{position}. {topic_name}"
            )

    except Exception as error:
        st.error(
            f"Connection error: {error}"
        )


def show_authenticated_app():
    """Display navigation for an authenticated learner."""

    profile_complete = st.session_state.get(
        "profile_complete",
        False,
    )

    if not profile_complete:
        st.sidebar.title("MindCompass")

        st.sidebar.write(
            "Complete your profile to continue."
        )

        if st.sidebar.button("Logout"):
            logout()

        render_onboarding()

        return

    st.sidebar.title("🧭 MindCompass")

    pages = [
        "Dashboard",
        "Onboarding",
        "Learning Path",
        "Progress",
    ]

    current_page = st.session_state.get(
        "page",
        "Dashboard",
    )

    if current_page not in pages:
        current_page = "Dashboard"

    selected_page = st.sidebar.selectbox(
        "Navigation",
        pages,
        index=pages.index(current_page),
    )

    st.session_state["page"] = selected_page

    if st.sidebar.button("Logout"):
        logout()
        return

    if selected_page == "Dashboard":
        render_dashboard()

    elif selected_page == "Onboarding":
        render_onboarding()

    elif selected_page == "Learning Path":
        show_learning_path()

    elif selected_page == "Progress":
        render_progress()


if "token" not in st.session_state:
    show_welcome()
else:
    show_authenticated_app()