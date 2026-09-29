"""Add a stable unique key to topics."""

from typing import Sequence, Union
import re

from alembic import op
import sqlalchemy as sa


revision: str = "365b1f5618f3"
down_revision: Union[str, Sequence[str], None] = "7eac5d5cf03f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def build_topic_key(topic_name: str) -> str:
    """
    Build a stable key from a topic name such as:
    Python | Ch8.1 – Functions -> python_ch8_1
    """
    match = re.match(
        r"^(.*?)\s*\|\s*Ch(\d+)\.(\d+)",
        topic_name,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot generate topic_key from topic name: {topic_name}"
        )

    course_name, chapter_number, lesson_number = match.groups()

    course_key = re.sub(
        r"[^a-z0-9]+",
        "_",
        course_name.strip().lower(),
    ).strip("_")

    return (
        f"{course_key}_ch"
        f"{int(chapter_number)}_{int(lesson_number)}"
    )


def upgrade() -> None:
    # Add the column as nullable while existing rows
    # are being assigned their stable keys.
    op.add_column(
        "topics",
        sa.Column(
            "topic_key",
            sa.String(length=100),
            nullable=True,
        ),
    )

    connection = op.get_bind()

    topics = connection.execute(
        sa.text(
            """
            SELECT topic_id, name
            FROM topics
            ORDER BY topic_id
            """
        )
    ).mappings().all()

    generated_keys: set[str] = set()

    for topic in topics:
        topic_key = build_topic_key(topic["name"])

        if topic_key in generated_keys:
            raise ValueError(
                f"Duplicate generated topic_key: {topic_key}"
            )

        generated_keys.add(topic_key)

        connection.execute(
            sa.text(
                """
                UPDATE topics
                SET topic_key = :topic_key
                WHERE topic_id = :topic_id
                """
            ),
            {
                "topic_key": topic_key,
                "topic_id": topic["topic_id"],
            },
        )

    # Require every topic to have a stable key.
    op.alter_column(
        "topics",
        "topic_key",
        existing_type=sa.String(length=100),
        nullable=False,
    )

    # Prevent two topics from sharing the same key.
    op.create_unique_constraint(
        "uq_topics_topic_key",
        "topics",
        ["topic_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_topics_topic_key",
        "topics",
        type_="unique",
    )

    op.drop_column(
        "topics",
        "topic_key",
    )