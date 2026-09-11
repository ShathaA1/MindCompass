from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
    Float,
    PrimaryKeyConstraint,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.database.connection import Base

def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    role = Column(String(50), nullable=False)

    learner_profile = relationship(
        "LearnerProfile",
        back_populates="user",
        uselist=False
    )

    learning_paths = relationship(
        "LearningPath",
        back_populates="user"
    )

    topic_masteries = relationship(
        "TopicMastery",
        back_populates="user"
    )

    assessment_attempts = relationship(
        "AssessmentAttempt",
        back_populates="user"
    )

    chat_sessions = relationship(
        "ChatSession",
        back_populates="user"
    )


class LearnerProfile(Base):
    __tablename__ = "learner_profile"

    profile_id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.user_id"),
        unique=True,
        nullable=False
    )

    goal = Column(String(255), nullable=False)
    initial_level = Column(String(20), nullable=False)
    weekly_hours = Column(Integer, nullable=False)
    current_level = Column(String(20), nullable=False)

    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )

    preferred_format = Column(String(50), nullable=False)
    preferred_pace = Column(String(20), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "initial_level IN ('beginner', 'intermediate', 'advanced')",
            name="chk_initial_level"
        ),
        CheckConstraint(
            "current_level IN ('beginner', 'intermediate', 'advanced')",
            name="chk_current_level"
        ),
        CheckConstraint(
            "preferred_pace IN ('slow', 'moderate', 'fast')",
            name="chk_preferred_pace"
        ),
        CheckConstraint(
            "weekly_hours > 0",
            name="chk_weekly_hours"
        ),
    )

    user = relationship(
        "User",
        back_populates="learner_profile"
    )


class Topic(Base):
    __tablename__ = "topics"

    topic_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(String(500), nullable=False)
    difficulty_level = Column(String(50), nullable=False)
    learning_objectives = Column(JSONB, nullable=False)

    learning_path_items = relationship(
        "LearningPathItem",
        back_populates="topic"
    )

    topic_masteries = relationship(
        "TopicMastery",
        back_populates="topic"
    )

    assessment_attempts = relationship(
        "AssessmentAttempt",
        back_populates="topic"
    )

    assessment_responses = relationship(
        "AssessmentResponse",
        back_populates="topic"
    )

    learning_materials = relationship(
        "LearningMaterial",
        back_populates="topic"
    )

    chat_sessions = relationship(
        "ChatSession",
        back_populates="topic"
    )


class TopicPrerequisite(Base):
    __tablename__ = "topic_prerequisites"

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=False
    )

    prerequisite_topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=False
    )

    __table_args__ = (
        PrimaryKeyConstraint(
            "topic_id",
            "prerequisite_topic_id"
        ),
    )


class LearningPath(Base):
    __tablename__ = "learning_paths"

    learning_path_id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.user_id"),
        nullable=False,
        index=True
    )

    name = Column(String(255), nullable=False)
    goal = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )

    user = relationship(
        "User",
        back_populates="learning_paths"
    )

    items = relationship(
        "LearningPathItem",
        back_populates="learning_path"
    )

    chat_sessions = relationship(
        "ChatSession",
        back_populates="learning_path"
    )


class LearningPathItem(Base):
    __tablename__ = "learning_path_items"

    learning_path_item_id = Column(Integer, primary_key=True)

    learning_path_id = Column(
        Integer,
        ForeignKey("learning_paths.learning_path_id"),
        nullable=False,
        index=True
    )

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=False
    )

    position = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)
    recommended_action = Column(String(20), nullable=False)

    started_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "learning_path_id",
            "position",
            name="uq_learning_path_position"
        ),
        UniqueConstraint(
            "learning_path_id",
            "topic_id",
            name="uq_learning_path_topic"
        ),
        CheckConstraint(
            '"position" > 0',
            name="chk_learning_path_position"
        ),
    )

    learning_path = relationship(
        "LearningPath",
        back_populates="items"
    )

    topic = relationship(
        "Topic",
        back_populates="learning_path_items"
    )


class TopicMastery(Base):
    __tablename__ = "topic_mastery"

    topic_mastery_id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.user_id"),
        nullable=False
    )

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=False
    )

    mastery_score = Column(Float, nullable=False)
    weak_areas = Column(JSONB, nullable=False)

    last_assessed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "topic_id",
            name="uq_topic_mastery_user_topic"
        ),
        CheckConstraint(
            "mastery_score BETWEEN 0 AND 100",
            name="chk_mastery_score"
        ),
    )

    user = relationship(
        "User",
        back_populates="topic_masteries"
    )

    topic = relationship(
        "Topic",
        back_populates="topic_masteries"
    )


class AssessmentAttempt(Base):
    __tablename__ = "assessment_attempts"

    assessment_attempt_id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.user_id"),
        nullable=False,
        index=True
    )

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=True
    )

    assessment_type = Column(String(30), nullable=False)
    score = Column(Float, nullable=False)
    max_score = Column(Float, nullable=False)
    feedback = Column(Text, nullable=True)

    started_at = Column(
        DateTime(timezone=True),
        nullable=False
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    __table_args__ = (
        CheckConstraint(
            "max_score > 0",
            name="chk_max_score"
        ),
        CheckConstraint(
            "score >= 0",
            name="chk_score"
        ),
        CheckConstraint(
            "score <= max_score",
            name="chk_score_not_above_max"
        ),
    )

    user = relationship(
        "User",
        back_populates="assessment_attempts"
    )

    topic = relationship(
        "Topic",
        back_populates="assessment_attempts"
    )

    responses = relationship(
        "AssessmentResponse",
        back_populates="attempt"
    )


class AssessmentResponse(Base):
    __tablename__ = "assessment_responses"

    assessment_response_id = Column(
        Integer,
        primary_key=True
    )

    attempt_id = Column(
        Integer,
        ForeignKey("assessment_attempts.assessment_attempt_id"),
        nullable=False
    )

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=False
    )

    question_text = Column(Text, nullable=False)
    question_type = Column(String(20), nullable=False)
    options = Column(JSONB, nullable=True)

    learner_answer = Column(Text, nullable=True)
    correct_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    score_awarded = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)

    attempt = relationship(
        "AssessmentAttempt",
        back_populates="responses"
    )

    topic = relationship(
        "Topic",
        back_populates="assessment_responses"
    )


class LearningMaterial(Base):
    __tablename__ = "learning_materials"

    learning_material_id = Column(
        Integer,
        primary_key=True
    )

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=False
    )

    title = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    source_path = Column(String(500), nullable=False)
    processing_status = Column(String(20), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    topic = relationship(
        "Topic",
        back_populates="learning_materials"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    chat_session_id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.user_id"),
        nullable=False,
        index=True
    )

    learning_path_id = Column(
        Integer,
        ForeignKey("learning_paths.learning_path_id"),
        nullable=True
    )

    topic_id = Column(
        Integer,
        ForeignKey("topics.topic_id"),
        nullable=True
    )

    session_name = Column(String(255), nullable=False)

    started_at = Column(
        DateTime(timezone=True),
        nullable=False
    )

    ended_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    user = relationship(
        "User",
        back_populates="chat_sessions"
    )

    learning_path = relationship(
        "LearningPath",
        back_populates="chat_sessions"
    )

    topic = relationship(
        "Topic",
        back_populates="chat_sessions"
    )

    messages = relationship(
        "ChatMessage",
        back_populates="session"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    chat_message_id = Column(Integer, primary_key=True)

    session_id = Column(
        Integer,
        ForeignKey("chat_sessions.chat_session_id"),
        nullable=False,
        index=True
    )

    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    agent_action = Column(String(20), nullable=True)
    retrieved_sources = Column(JSONB, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            """
            agent_action IS NULL
            OR agent_action IN (
                'explain',
                'assess',
                'practice',
                'review',
                'recommend'
            )
            """,
            name="chk_agent_action"
        ),
    )

    session = relationship(
        "ChatSession",
        back_populates="messages"
    )