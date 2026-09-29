from app.core.learning_paths import AVAILABLE_LEARNING_PATHS
from app.database.connection import SessionLocal
from app.services.learning_service import (
    get_diagnostic_topics,
    get_topic_id_by_key,
)


def test_learning_path_targets_resolve_from_keys():
    with SessionLocal() as db:
        for config in AVAILABLE_LEARNING_PATHS.values():
            topic_id = get_topic_id_by_key(
                db,
                config["target_topic_key"],
            )
            assert isinstance(topic_id, int)


def test_diagnostic_topics_resolve_from_keys():
    with SessionLocal() as db:
        topics = get_diagnostic_topics(db, "agentic_ai")

        assert [topic["topic_id"] for topic in topics] == [
            get_topic_id_by_key(db, "python_ch0_1"),
            get_topic_id_by_key(db, "machine_learning_ch0_1"),
        ]
