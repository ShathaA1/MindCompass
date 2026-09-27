"""Defines request and response schemas for Tutor chat interactions."""

from typing import Any

from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    """Request data for starting a new Tutor chat session."""

    session_name: str = Field(
        default="Tutor Session",
        min_length=1,
        max_length=255,
    )


class ChatMessageRequest(BaseModel):
    """Request data for sending a message to the Tutor Agent."""

    message: str = Field(
        min_length=1,
    )


class AssessmentAnswer(BaseModel):
    """A learner answer submitted for one assessment question."""

    learner_answer: str = Field(
        min_length=1,
    )


class AssessmentSubmissionRequest(BaseModel):
    """
    Request data for submitting answers to a
    generated Tutor assessment.
    """

    assessment_type: str = "topic"

    assessment_questions: list[dict[str, Any]]

    assessment_answers: list[AssessmentAnswer]