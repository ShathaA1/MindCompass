"""Defines schemas for assessments, questions, answers, and feedback."""

from pydantic import BaseModel


class DiagnosticStartRequest(BaseModel):
    """
    Request body used to start the initial diagnostic
    for the learner's selected learning path.
    """

    selected_path: str