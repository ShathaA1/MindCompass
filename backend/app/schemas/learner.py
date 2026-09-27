from typing import Literal
from pydantic import BaseModel, Field

LearningStyle = Literal[
    "Concise Explanations",
    "Detailed Explanations",
    "Practical Examples",
    "Guided Practice",
]
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class OnboardingRequest(BaseModel):
    goal: str
    initial_level: Literal["beginner", "intermediate", "advanced"]
    weekly_hours: int = Field(gt=0)
    preferred_format: LearningStyle
    preferred_pace: Literal["slow", "moderate", "fast"]
