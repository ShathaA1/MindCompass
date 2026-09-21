"""Defines schemas for assessments, questions, answers, and feedback."""
"""Defines schemas for assessments, questions, answers, and feedback."""

from pydantic import BaseModel


class DiagnosticStartRequest(BaseModel):
    """
    Request body used to start the initial diagnostic
    for the learner's selected learning path.
    """

    selected_path: str


class DiagnosticAnswer(BaseModel):
    """
    Represents one learner answer submitted
    for an initial diagnostic question.
    """

    learner_answer: str


class DiagnosticSubmitRequest(BaseModel):
    """
    Request body used to submit the learner's
    completed initial diagnostic.
    """

    selected_path: str
    assessment_questions: list[dict]
    assessment_answers: list[DiagnosticAnswer]