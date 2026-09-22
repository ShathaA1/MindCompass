"""Defines request and response schemas for Tutor chat interactions."""

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