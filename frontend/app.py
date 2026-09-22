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

    mode = st.session_state.get("welcome_mode")

    # Display the landing page when no authentication
    # form is currently selected.
    if mode is None:
        st.markdown(
            '<div class="mc-welcome-page">',
            unsafe_allow_html=True,
        )

        # Display the MindCompass logo.
        logo_col_left, logo_col, logo_col_right = st.columns(
            [1, 1.2, 1]
        )

        with logo_col:
            st.image(
                "frontend/assets/mindcompass_logo.png",
                width="stretch",
            )

        # Main welcome content.
        welcome_html = (
            '<div class="mc-welcome-content">'
            '<h1>MindCompass</h1>'
            '<h2>Your Personalized AI Tutor</h2>'
            '<p class="mc-welcome-description">'
            'A learning experience that understands your goals, '
            'adapts to your level, and guides you through every '
            'step of your learning journey.'
            '</p>'
            '</div>'
        )

        st.markdown(
            welcome_html,
            unsafe_allow_html=True,
        )

        # Keep the main actions centered.
        left_space, action_col, right_space = st.columns(
            [1, 1.2, 1]
        )

        with action_col:
            if st.button(
                "Login",
                type="primary",
                width="stretch",
            ):
                st.session_state["welcome_mode"] = "Login"
                rerun_app()

            if st.button(
                "Create Account",
                width="stretch",
            ):
                st.session_state["welcome_mode"] = "Register"
                rerun_app()

        st.markdown(
            """
            <p class="mc-welcome-footer">
                Learn at your pace. Adapt as you grow.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # Display the login form.
    elif mode == "Login":
        if st.button("← Back"):
            del st.session_state["welcome_mode"]
            rerun_app()

        show_login()

    # Display the registration form.
    elif mode == "Register":
        if st.button("← Back"):
            del st.session_state["welcome_mode"]
            rerun_app()

        show_register()


def show_login():
    """Display the login form."""

    # Center the login form on the page.
    left_space, form_col, right_space = st.columns(
        [1, 1.2, 1]
    )

    with form_col:
        st.markdown(
            (
                '<div class="mc-auth-header">'
                '<h1>Welcome back</h1>'
                '<p>Sign in to continue your learning journey.</p>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        email = st.text_input(
            "Email",
            placeholder="Enter your email",
            key="login_email",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password",
        )

        if st.button(
            "Login",
            type="primary",
            width="stretch",
        ):
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
                        st.session_state[
                            "profile_complete"
                        ] = True

                        # Check whether the learner already has
                        # a personalized learning path.
                        path_response = get_learning_path(
                            token
                        )

                        if (
                            path_response.status_code == 200
                            and path_response.json().get(
                                "learning_path"
                            ) is not None
                        ):
                            # Existing learners with a learning path
                            # continue directly to the main application.
                            st.session_state[
                                "setup_complete"
                            ] = True

                            st.session_state[
                                "page"
                            ] = "Dashboard"

                        else:
                            # Learners without a learning path must
                            # complete the learning setup first.
                            st.session_state[
                                "setup_complete"
                            ] = False

                            st.session_state[
                                "page"
                            ] = "Diagnostic"

                    elif profile_response.status_code == 404:
                        st.session_state[
                            "profile_complete"
                        ] = False

                        st.session_state[
                            "setup_complete"
                        ] = False

                        st.session_state[
                            "page"
                        ] = "Onboarding"

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

        st.markdown(
            '<p class="mc-auth-switch-text">'
            "Don't have an account?"
            '</p>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Create Account",
            width="stretch",
            key="login_create_account",
        ):
            st.session_state[
                "welcome_mode"
            ] = "Register"

            rerun_app()


def show_register():
    """Display the registration form."""

    # Center the registration form on the page.
    left_space, form_col, right_space = st.columns(
        [1, 1.2, 1]
    )

    with form_col:
        st.markdown(
            (
                '<div class="mc-auth-header">'
                '<h1>Create your account</h1>'
                '<p>Start your personalized learning journey.</p>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        name = st.text_input(
            "Name",
            placeholder="Enter your name",
            key="register_name",
        )

        email = st.text_input(
            "Email",
            placeholder="Enter your email",
            key="register_email",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Create a password",
            key="register_password",
        )

        if st.button(
            "Create Account",
            type="primary",
            width="stretch",
        ):
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

                    st.session_state[
                        "welcome_mode"
                    ] = "Login"

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

        st.markdown(
            '<p class="mc-auth-switch-text">'
            'Already have an account?'
            '</p>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Login",
            width="stretch",
            key="register_login",
        ):
            st.session_state[
                "welcome_mode"
            ] = "Login"

            rerun_app()


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
        width="stretch",
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
        width="stretch",
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
            width="stretch",
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
        width="stretch",
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
