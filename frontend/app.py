"""Configures the main Streamlit application and learner navigation."""

import streamlit as st
from styles import apply_global_styles

from api_client import get_profile, get_learning_path, login, register
from pages.dashboard import render as render_dashboard
from pages.onboarding import render as render_onboarding
from pages.diagnostic import render as render_diagnostic
from pages.tutor import render as render_tutor

st.set_page_config(
    page_title="MindCompass",
    page_icon="🧭",
    layout="wide",
)

apply_global_styles()

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
    """Clear all data stored for the current learner session."""

    # Clear authentication, navigation, and learning setup data.
    for key in [
        "token",
        "page",
        "profile_complete",
        "setup_complete",
        "welcome_mode",
        "selected_path",
        "diagnostic_topics",
        "diagnostic_questions",
        "diagnostic_started",
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
            if st.button(
                "Login",
                type="primary",
                use_container_width=True,
            ):
                st.session_state["welcome_mode"] = "Login"
                rerun_app()

        with register_col:
            if st.button(
                "Create Account",
                use_container_width=True,
            ):
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

                    # Check whether the learner already has
                    # a personalized learning path.
                    path_response = get_learning_path(token)

                    if (
                        path_response.status_code == 200
                        and path_response.json().get("learning_path") is not None
                    ):
                        # Existing learners with a learning path
                        # continue directly to the main application.
                        st.session_state["setup_complete"] = True
                        st.session_state["page"] = "Dashboard"

                    else:
                        # Learners without a learning path must complete
                        # learning path selection and the diagnostic first.
                        st.session_state["setup_complete"] = False
                        st.session_state["page"] = "Diagnostic"

                elif profile_response.status_code == 404:
                    st.session_state["profile_complete"] = False
                    st.session_state["setup_complete"] = False
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


def render_main_sidebar():
    """Render the main MindCompass sidebar navigation."""

    st.sidebar.title("MindCompass")
    st.sidebar.caption("Your AI Tutor")

    st.sidebar.markdown("### MAIN")

    pages = [
        "Dashboard",
        "Tutor",
        "Learning Path",
    ]

    current_page = st.session_state.get(
        "page",
        "Dashboard",
    )

    # Use Dashboard as the default page.
    if current_page not in pages:
        current_page = "Dashboard"

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(current_page),
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "Logout",
        use_container_width=True,
    ):
        logout()
        return None

    return selected_page


def render_setup_sidebar():
    """Render the sidebar while the learner completes setup."""

    st.sidebar.title("MindCompass")
    st.sidebar.caption("Your AI Tutor")

    st.sidebar.markdown("### MAIN")

    # Display the main pages as disabled during setup.
    st.sidebar.markdown(
        """
        <div class="mc-disabled-nav">
            <div>Dashboard</div>
            <div>Tutor</div>
            <div>Learning Path</div>
        </div>

        <div class="mc-setup-note">
            Complete your learning setup to access these pages.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    if st.sidebar.button(
        "Logout",
        use_container_width=True,
    ):
        logout()
        return False

    return True



def show_authenticated_app():
    """Display navigation for an authenticated learner."""

    profile_complete = st.session_state.get(
        "profile_complete",
        False,
    )

    # Keep onboarding separate from the main application.
    if not profile_complete:
        st.sidebar.title("MindCompass")

        st.sidebar.write(
            "Complete your profile to continue."
        )

        if st.sidebar.button(
            "Logout",
            use_container_width=True,
        ):
            logout()
            return

        render_onboarding()
        return

    # Keep the diagnostic setup flow separate
    # from the main application.
    setup_complete = st.session_state.get(
        "setup_complete",
        False,
    )

    if not setup_complete:
        if not render_setup_sidebar():
            return

        render_diagnostic()
        return

    # Main application navigation.
    st.sidebar.title("🧭 MindCompass")
    st.sidebar.caption("Your AI Tutor")

    st.sidebar.markdown("### MAIN")

    pages = [
        "Dashboard",
        "Tutor",
        "Learning Path",
    ]

    page_names = {
        "Dashboard": "Dashboard",
        "Tutor": "Tutor",
        "Learning Path": "Learning Path",
    }

    current_page = st.session_state.get(
        "page",
        "Dashboard",
    )

    # Find the sidebar label that matches
    # the currently active page.
    current_label = next(
        (
            label
            for label, page_name in page_names.items()
            if page_name == current_page
        ),
        "Dashboard",
    )

    selected_label = st.sidebar.radio(
        "Navigation",
        pages,
        index=pages.index(current_label),
        label_visibility="collapsed",
    )

    selected_page = page_names[selected_label]

    st.session_state["page"] = selected_page

    st.sidebar.divider()

    if st.sidebar.button(
        "Logout",
        use_container_width=True,
    ):
        logout()
        return

    # Render the selected main application page.
    if selected_page == "Dashboard":
        render_dashboard()

    elif selected_page == "Tutor":
        render_tutor()

    elif selected_page == "Learning Path":
        show_learning_path()


if "token" not in st.session_state:
    show_welcome()
else:
    show_authenticated_app()