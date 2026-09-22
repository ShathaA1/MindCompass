"""Sends requests from the Streamlit interface to the FastAPI backend."""

import os

import requests
from dotenv import load_dotenv


load_dotenv()

BASE_URL = os.getenv(
    "FASTAPI_BASE_URL",
    "http://localhost:8000",
)

REQUEST_TIMEOUT = 90


def _headers(token=None):
    """Create authorization headers when a token is available."""

    if token:
        return {
            "Authorization": f"Bearer {token}"
        }

    return {}


def register(name, email, password):
    """Register a new learner."""

    return requests.post(
        f"{BASE_URL}/auth/register",
        json={
            "name": name,
            "email": email,
            "password": password,
        },
        timeout=REQUEST_TIMEOUT,
    )


def login(email, password):
    """Authenticate a learner."""

    return requests.post(
        f"{BASE_URL}/auth/login",
        json={
            "email": email,
            "password": password,
        },
        timeout=REQUEST_TIMEOUT,
    )


def get_me(token):
    """Get the authenticated learner."""

    return requests.get(
        f"{BASE_URL}/auth/me",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
    )


def create_onboarding(token, data):
    """Create the learner profile."""

    return requests.post(
        f"{BASE_URL}/learners/onboarding",
        headers=_headers(token),
        json=data,
        timeout=REQUEST_TIMEOUT,
    )


def get_profile(token):
    """Get the learner profile."""

    return requests.get(
        f"{BASE_URL}/learners/profile",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
    )


def update_profile(token, data):
    """Update the learner profile."""

    return requests.put(
        f"{BASE_URL}/learners/profile",
        headers=_headers(token),
        json=data,
        timeout=REQUEST_TIMEOUT,
    )


def get_dashboard(token):
    """Get dashboard data."""

    return requests.get(
        f"{BASE_URL}/learning/dashboard",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
    )


def get_progress(token):
    """Get learner progress."""

    return requests.get(
        f"{BASE_URL}/learning/progress",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
    )


def get_learning_path(token):
    """Get the learner's learning path."""

    return requests.get(
        f"{BASE_URL}/learning/learning-path",
        headers=_headers(token),
        timeout=REQUEST_TIMEOUT,
    )

def start_diagnostic(token, selected_path):
    """
    Start the initial diagnostic for the learner's
    selected learning path.
    """

    return requests.post(
        f"{BASE_URL}/learning/diagnostic/start",
        headers=_headers(token),
        json={
            "selected_path": selected_path,
        },
        timeout=REQUEST_TIMEOUT,
    )


def submit_diagnostic(
    token,
    selected_path,
    assessment_questions,
    assessment_answers,
):
    """
    Submit the learner's initial diagnostic answers
    to create the personalized learning path.
    """

    return requests.post(
        f"{BASE_URL}/learning/diagnostic/submit",
        headers=_headers(token),
        json={
            "selected_path": selected_path,
            "assessment_questions": assessment_questions,
            "assessment_answers": assessment_answers,
        },
        timeout=REQUEST_TIMEOUT,
    )


def create_chat_session(
    token,
    session_name="Tutor Session"
):
    """
    Create a new Tutor chat session for
    the authenticated learner.
    """

    return requests.post(
        f"{BASE_URL}/chat/sessions",
        headers=_headers(token),
        json={
            "session_name": session_name,
        },
        timeout=REQUEST_TIMEOUT,
    )


def send_chat_message(
    token,
    session_id,
    message
):
    """
    Send a learner message to the Tutor Agent
    and return the generated Tutor response.
    """

    return requests.post(
        f"{BASE_URL}/chat/{session_id}/messages",
        headers=_headers(token),
        json={
            "message": message,
        },
        timeout=REQUEST_TIMEOUT,
    )