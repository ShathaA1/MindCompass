from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "7eac5d5cf03f"
down_revision: Union[str, Sequence[str], None] = "64c0ec5c8d67"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename user table instead of dropping/recreating it
    op.rename_table("user", "users")

    # Remove redundant PK indexes
    op.drop_index("ix_user_user_id", table_name="users")
    op.drop_index(
        "ix_assessment_attempts_assessment_attempt_id",
        table_name="assessment_attempts"
    )
    op.drop_index(
        "ix_assessment_responses_assessment_response_id",
        table_name="assessment_responses"
    )
    op.drop_index(
        "ix_chat_messages_chat_message_id",
        table_name="chat_messages"
    )
    op.drop_index(
        "ix_chat_sessions_chat_session_id",
        table_name="chat_sessions"
    )
    op.drop_index(
        "ix_learner_profile_profile_id",
        table_name="learner_profile"
    )
    op.drop_index(
        "ix_learning_materials_learning_material_id",
        table_name="learning_materials"
    )
    op.drop_index(
        "ix_learning_path_items_learning_path_item_id",
        table_name="learning_path_items"
    )
    op.drop_index(
        "ix_learning_paths_learning_path_id",
        table_name="learning_paths"
    )
    op.drop_index(
        "ix_topic_mastery_topic_mastery_id",
        table_name="topic_mastery"
    )
    op.drop_index(
        "ix_topics_topic_id",
        table_name="topics"
    )

    # Rename existing email index to match users table
    op.execute(
        "ALTER INDEX ix_user_email RENAME TO ix_users_email"
    )

    # Add required indexes
    op.create_index(
        "ix_assessment_attempts_user_id",
        "assessment_attempts",
        ["user_id"],
        unique=False
    )
    op.create_index(
        "ix_chat_messages_session_id",
        "chat_messages",
        ["session_id"],
        unique=False
    )
    op.create_index(
        "ix_chat_sessions_user_id",
        "chat_sessions",
        ["user_id"],
        unique=False
    )
    op.create_index(
        "ix_learning_path_items_learning_path_id",
        "learning_path_items",
        ["learning_path_id"],
        unique=False
    )
    op.create_index(
        "ix_learning_paths_user_id",
        "learning_paths",
        ["user_id"],
        unique=False
    )

    # Add learning path name
    op.add_column(
        "learning_paths",
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False
        )
    )

    # Unique constraints
    op.create_unique_constraint(
        "uq_learning_path_position",
        "learning_path_items",
        ["learning_path_id", "position"]
    )
    op.create_unique_constraint(
        "uq_learning_path_topic",
        "learning_path_items",
        ["learning_path_id", "topic_id"]
    )
    op.create_unique_constraint(
        "uq_topic_mastery_user_topic",
        "topic_mastery",
        ["user_id", "topic_id"]
    )

    # Check constraints
    op.create_check_constraint(
        "chk_initial_level",
        "learner_profile",
        "initial_level IN ('beginner', 'intermediate', 'advanced')"
    )
    op.create_check_constraint(
        "chk_current_level",
        "learner_profile",
        "current_level IN ('beginner', 'intermediate', 'advanced')"
    )
    op.create_check_constraint(
        "chk_preferred_pace",
        "learner_profile",
        "preferred_pace IN ('slow', 'moderate', 'fast')"
    )
    op.create_check_constraint(
        "chk_weekly_hours",
        "learner_profile",
        "weekly_hours > 0"
    )
    op.create_check_constraint(
        "chk_mastery_score",
        "topic_mastery",
        "mastery_score BETWEEN 0 AND 100"
    )
    op.create_check_constraint(
        "chk_learning_path_position",
        "learning_path_items",
        '"position" > 0'
    )
    op.create_check_constraint(
        "chk_max_score",
        "assessment_attempts",
        "max_score > 0"
    )
    op.create_check_constraint(
        "chk_score",
        "assessment_attempts",
        "score >= 0"
    )
    op.create_check_constraint(
        "chk_score_not_above_max",
        "assessment_attempts",
        "score <= max_score"
    )
    op.create_check_constraint(
        "chk_agent_action",
        "chat_messages",
        """
        agent_action IS NULL
        OR agent_action IN (
            'explain',
            'assess',
            'practice',
            'review',
            'recommend'
        )
        """
    )

    # JSON -> JSONB
    op.alter_column(
        "assessment_responses",
        "options",
        existing_type=postgresql.JSON(),
        type_=postgresql.JSONB(),
        postgresql_using="options::jsonb",
        existing_nullable=True
    )
    op.alter_column(
        "chat_messages",
        "retrieved_sources",
        existing_type=postgresql.JSON(),
        type_=postgresql.JSONB(),
        postgresql_using="retrieved_sources::jsonb",
        existing_nullable=True
    )
    op.alter_column(
        "topic_mastery",
        "weak_areas",
        existing_type=postgresql.JSON(),
        type_=postgresql.JSONB(),
        postgresql_using="weak_areas::jsonb",
        existing_nullable=False
    )
    op.alter_column(
        "topics",
        "learning_objectives",
        existing_type=postgresql.JSON(),
        type_=postgresql.JSONB(),
        postgresql_using="learning_objectives::jsonb",
        existing_nullable=False
    )

    # Timestamp without timezone -> timestamp with timezone
    timestamp_columns = [
        ("assessment_attempts", "started_at", False),
        ("assessment_attempts", "completed_at", True),
        ("chat_messages", "created_at", False),
        ("chat_sessions", "started_at", False),
        ("chat_sessions", "ended_at", True),
        ("learner_profile", "updated_at", False),
        ("learning_materials", "created_at", False),
        ("learning_path_items", "started_at", True),
        ("learning_path_items", "completed_at", True),
        ("learning_paths", "created_at", False),
        ("learning_paths", "updated_at", False),
        ("topic_mastery", "last_assessed_at", True),
        ("users", "created_at", False),
    ]

    for table_name, column_name, nullable in timestamp_columns:
        op.alter_column(
            table_name,
            column_name,
            existing_type=postgresql.TIMESTAMP(),
            type_=sa.DateTime(timezone=True),
            postgresql_using=f"{column_name} AT TIME ZONE 'UTC'",
            existing_nullable=nullable
        )


def downgrade() -> None:
    timestamp_columns = [
        ("assessment_attempts", "started_at", False),
        ("assessment_attempts", "completed_at", True),
        ("chat_messages", "created_at", False),
        ("chat_sessions", "started_at", False),
        ("chat_sessions", "ended_at", True),
        ("learner_profile", "updated_at", False),
        ("learning_materials", "created_at", False),
        ("learning_path_items", "started_at", True),
        ("learning_path_items", "completed_at", True),
        ("learning_paths", "created_at", False),
        ("learning_paths", "updated_at", False),
        ("topic_mastery", "last_assessed_at", True),
        ("users", "created_at", False),
    ]

    for table_name, column_name, nullable in reversed(timestamp_columns):
        op.alter_column(
            table_name,
            column_name,
            existing_type=sa.DateTime(timezone=True),
            type_=postgresql.TIMESTAMP(),
            postgresql_using=f"{column_name} AT TIME ZONE 'UTC'",
            existing_nullable=nullable
        )

    op.alter_column(
        "topics",
        "learning_objectives",
        existing_type=postgresql.JSONB(),
        type_=postgresql.JSON(),
        postgresql_using="learning_objectives::json",
        existing_nullable=False
    )
    op.alter_column(
        "topic_mastery",
        "weak_areas",
        existing_type=postgresql.JSONB(),
        type_=postgresql.JSON(),
        postgresql_using="weak_areas::json",
        existing_nullable=False
    )
    op.alter_column(
        "chat_messages",
        "retrieved_sources",
        existing_type=postgresql.JSONB(),
        type_=postgresql.JSON(),
        postgresql_using="retrieved_sources::json",
        existing_nullable=True
    )
    op.alter_column(
        "assessment_responses",
        "options",
        existing_type=postgresql.JSONB(),
        type_=postgresql.JSON(),
        postgresql_using="options::json",
        existing_nullable=True
    )

    op.drop_constraint(
        "chk_agent_action",
        "chat_messages",
        type_="check"
    )
    op.drop_constraint(
        "chk_score_not_above_max",
        "assessment_attempts",
        type_="check"
    )
    op.drop_constraint(
        "chk_score",
        "assessment_attempts",
        type_="check"
    )
    op.drop_constraint(
        "chk_max_score",
        "assessment_attempts",
        type_="check"
    )
    op.drop_constraint(
        "chk_learning_path_position",
        "learning_path_items",
        type_="check"
    )
    op.drop_constraint(
        "chk_mastery_score",
        "topic_mastery",
        type_="check"
    )
    op.drop_constraint(
        "chk_weekly_hours",
        "learner_profile",
        type_="check"
    )
    op.drop_constraint(
        "chk_preferred_pace",
        "learner_profile",
        type_="check"
    )
    op.drop_constraint(
        "chk_current_level",
        "learner_profile",
        type_="check"
    )
    op.drop_constraint(
        "chk_initial_level",
        "learner_profile",
        type_="check"
    )

    op.drop_constraint(
        "uq_topic_mastery_user_topic",
        "topic_mastery",
        type_="unique"
    )
    op.drop_constraint(
        "uq_learning_path_topic",
        "learning_path_items",
        type_="unique"
    )
    op.drop_constraint(
        "uq_learning_path_position",
        "learning_path_items",
        type_="unique"
    )

    op.drop_column(
        "learning_paths",
        "name"
    )

    op.drop_index(
        "ix_learning_paths_user_id",
        table_name="learning_paths"
    )
    op.drop_index(
        "ix_learning_path_items_learning_path_id",
        table_name="learning_path_items"
    )
    op.drop_index(
        "ix_chat_sessions_user_id",
        table_name="chat_sessions"
    )
    op.drop_index(
        "ix_chat_messages_session_id",
        table_name="chat_messages"
    )
    op.drop_index(
        "ix_assessment_attempts_user_id",
        table_name="assessment_attempts"
    )

    # Restore old redundant PK indexes
    op.create_index(
        "ix_topics_topic_id",
        "topics",
        ["topic_id"],
        unique=False
    )
    op.create_index(
        "ix_topic_mastery_topic_mastery_id",
        "topic_mastery",
        ["topic_mastery_id"],
        unique=False
    )
    op.create_index(
        "ix_learning_paths_learning_path_id",
        "learning_paths",
        ["learning_path_id"],
        unique=False
    )
    op.create_index(
        "ix_learning_path_items_learning_path_item_id",
        "learning_path_items",
        ["learning_path_item_id"],
        unique=False
    )
    op.create_index(
        "ix_learning_materials_learning_material_id",
        "learning_materials",
        ["learning_material_id"],
        unique=False
    )
    op.create_index(
        "ix_learner_profile_profile_id",
        "learner_profile",
        ["profile_id"],
        unique=False
    )
    op.create_index(
        "ix_chat_sessions_chat_session_id",
        "chat_sessions",
        ["chat_session_id"],
        unique=False
    )
    op.create_index(
        "ix_chat_messages_chat_message_id",
        "chat_messages",
        ["chat_message_id"],
        unique=False
    )
    op.create_index(
        "ix_assessment_responses_assessment_response_id",
        "assessment_responses",
        ["assessment_response_id"],
        unique=False
    )
    op.create_index(
        "ix_assessment_attempts_assessment_attempt_id",
        "assessment_attempts",
        ["assessment_attempt_id"],
        unique=False
    )
    op.create_index(
        "ix_user_user_id",
        "users",
        ["user_id"],
        unique=False
    )

    op.execute(
        "ALTER INDEX ix_users_email RENAME TO ix_user_email"
    )

    op.rename_table(
        "users",
        "user"
    )