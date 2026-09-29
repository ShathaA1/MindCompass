"""Add or update topics and prerequisites from the knowledge base."""

import argparse
import json
import re
import sys
from pathlib import Path


_BACKEND_DIR = str(
    Path(__file__).resolve().parent.parent / "backend"
)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from sqlalchemy.orm import Session

from backend.app.database.connection import SessionLocal
from backend.app.database.models import Topic, TopicPrerequisite


DEFAULT_DIFFICULTY_LEVEL = "intermediate"

TOPIC_FOLDER_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "app"
    / "rag"
    / "topic_folder_map.json"
)


def _clean_title(filename: str) -> str:
    """Turn a slide or lab filename into a readable phrase."""
    stem = Path(filename).stem
    stem = re.sub(
        r"^(slides_|lab_)",
        "",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"\(.*?\)", "", stem)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return re.sub(r"\s+", " ", stem)


def _title_to_objective(
    title: str,
    *,
    is_practical: bool,
) -> str:
    """Turn a cleaned title into a learning objective."""
    first_word = title.split(" ", 1)[0] if title else ""

    if first_word.isupper() and len(first_word) > 1:
        lowered = title
    else:
        lowered = title[0].lower() + title[1:] if title else title

    if is_practical:
        return f"Apply {lowered} through hands-on practice"

    return f"Understand {lowered}"


def _extract_learning_objectives(
    lesson_dir: Path,
) -> list[str]:
    """Derive objectives from slide titles, falling back to labs."""
    slide_files = sorted(lesson_dir.glob("slides_*.pdf"))
    source_files = slide_files or sorted(
        lesson_dir.glob("lab_*.ipynb")
    )
    is_practical = not slide_files

    objectives = []
    seen = set()

    for source_file in source_files:
        title = _clean_title(source_file.name)

        if not title:
            continue

        objective = _title_to_objective(
            title,
            is_practical=is_practical,
        )

        if objective.lower() not in seen:
            objectives.append(objective)
            seen.add(objective.lower())

    return objectives


def _lesson_name_from_folder(folder_name: str) -> str:
    """Remove the numeric prefix and tidy underscores."""
    name = re.sub(
        r"^\d{2}_\d{2}_",
        "",
        folder_name,
    ).strip()
    name = re.sub(r"[_]+", " ", name)
    return re.sub(r"\s+", " ", name).strip(" :")


def _chapter_number_from_folder(folder_name: str) -> int:
    match = re.match(r"^(\d{2})_", folder_name)
    return int(match.group(1)) if match else 0


def build_topic_key(
    course_name: str,
    chapter_number: int,
    lesson_number: int,
) -> str:
    """Build a stable key independent of the database topic ID."""
    course_key = re.sub(
        r"[^a-z0-9]+",
        "_",
        course_name.strip().lower(),
    ).strip("_")

    return (
        f"{course_key}_ch"
        f"{chapter_number}_{lesson_number}"
    )


def _is_chapter_folder(folder_name: str) -> bool:
    """Return whether a folder name starts with NN_."""
    return re.match(r"^\d{2}_", folder_name) is not None


def _build_lesson(
    *,
    course_name: str,
    chapter_number: int,
    lesson_number: int,
    lesson_dir: Path,
    lesson_name: str,
    description: str,
) -> dict:
    """Build one normalized lesson record."""
    return {
        "course_name": course_name,
        "chapter_number": chapter_number,
        "lesson_number": lesson_number,
        "topic_key": build_topic_key(
            course_name,
            chapter_number,
            lesson_number,
        ),
        "lesson_dir": lesson_dir,
        "name": (
            f"{course_name} | "
            f"Ch{chapter_number}.{lesson_number} – "
            f"{lesson_name}"
        ),
        "description": description,
        "learning_objectives": _extract_learning_objectives(
            lesson_dir
        ),
    }


def discover_lessons(source_root: Path) -> list[dict]:
    """Discover lessons in curriculum order."""
    lessons = []
    top_level_dirs = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir()
    )

    for entry in top_level_dirs:
        if _is_chapter_folder(entry.name):
            course_name = source_root.name
            chapter_dirs = [entry]
        elif not any(path.is_dir() for path in entry.iterdir()):
            course_name = entry.name
            lesson_name = "Reference Material"
            lessons.append(
                _build_lesson(
                    course_name=course_name,
                    chapter_number=0,
                    lesson_number=1,
                    lesson_dir=entry,
                    lesson_name=lesson_name,
                    description=(
                        f"Course: {course_name} | Supplementary "
                        "reference material (not part of the "
                        "sequenced curriculum)."
                    ),
                )
            )
            continue
        else:
            course_name = entry.name
            chapter_dirs = sorted(
                path
                for path in entry.iterdir()
                if path.is_dir()
            )

        for chapter_dir in chapter_dirs:
            chapter_number = _chapter_number_from_folder(
                chapter_dir.name
            )
            lesson_dirs = sorted(
                path
                for path in chapter_dir.iterdir()
                if path.is_dir()
            )

            if not lesson_dirs:
                chapter_title = re.sub(
                    r"^\d{2}_(Chapter_\d+_)?",
                    "",
                    chapter_dir.name,
                )
                chapter_title = re.sub(
                    r"[_]+",
                    " ",
                    chapter_title,
                ).strip()
                lessons.append(
                    _build_lesson(
                        course_name=course_name,
                        chapter_number=chapter_number,
                        lesson_number=1,
                        lesson_dir=chapter_dir,
                        lesson_name=chapter_title,
                        description=(
                            f"Course: {course_name} | Chapter "
                            f"{chapter_number} lesson: "
                            f"{chapter_title}"
                        ),
                    )
                )
                continue

            for lesson_dir in lesson_dirs:
                lesson_name = _lesson_name_from_folder(
                    lesson_dir.name
                )
                lesson_number = int(
                    lesson_dir.name.split("_")[1]
                )
                lessons.append(
                    _build_lesson(
                        course_name=course_name,
                        chapter_number=chapter_number,
                        lesson_number=lesson_number,
                        lesson_dir=lesson_dir,
                        lesson_name=lesson_name,
                        description=(
                            f"Course: {course_name} | Chapter "
                            f"{chapter_number} lesson: {lesson_name}"
                        ),
                    )
                )

    return lessons


def seed_topics(
    session: Session,
    source_root: Path,
) -> list[Topic]:
    """Insert or update topics and their prerequisite sequence."""
    lessons = discover_lessons(source_root)
    existing_topics = session.query(Topic).all()

    existing_topics_by_key = {
        topic.topic_key: topic
        for topic in existing_topics
        if topic.topic_key
    }
    existing_topics_by_name = {
        topic.name: topic
        for topic in existing_topics
    }

    all_topics: list[Topic] = []
    folder_map_entries: list[dict] = []
    previous_topic_by_course: dict[str, Topic] = {}

    for lesson in lessons:
        topic_key = lesson["topic_key"]
        topic = existing_topics_by_key.get(topic_key)

        if topic is None:
            topic = existing_topics_by_name.get(lesson["name"])

        if topic is None:
            topic = Topic(
                topic_key=topic_key,
                name=lesson["name"],
                description=lesson["description"],
                difficulty_level=DEFAULT_DIFFICULTY_LEVEL,
                learning_objectives=lesson[
                    "learning_objectives"
                ],
            )
            session.add(topic)
            session.flush()
        else:
            topic.topic_key = topic_key
            topic.name = lesson["name"]
            topic.description = lesson["description"]
            topic.learning_objectives = lesson[
                "learning_objectives"
            ]

        existing_topics_by_key[topic_key] = topic
        existing_topics_by_name[lesson["name"]] = topic

        course_name = lesson["course_name"]
        previous_topic = previous_topic_by_course.get(
            course_name
        )

        if previous_topic is not None:
            already_linked = (
                session.query(TopicPrerequisite)
                .filter_by(
                    topic_id=topic.topic_id,
                    prerequisite_topic_id=(
                        previous_topic.topic_id
                    ),
                )
                .first()
            )

            if already_linked is None:
                session.add(
                    TopicPrerequisite(
                        topic_id=topic.topic_id,
                        prerequisite_topic_id=(
                            previous_topic.topic_id
                        ),
                    )
                )

        previous_topic_by_course[course_name] = topic
        all_topics.append(topic)
        folder_map_entries.append(
            {
                "folder_path": str(
                    lesson["lesson_dir"].resolve()
                ),
                "topic_id": topic.topic_id,
                "topic_key": topic.topic_key,
                "course_name": course_name,
                "chapter_number": lesson["chapter_number"],
                "lesson_number": lesson["lesson_number"],
                "lesson_name": lesson["name"],
            }
        )

    session.commit()

    TOPIC_FOLDER_MAP_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    TOPIC_FOLDER_MAP_PATH.write_text(
        json.dumps(
            folder_map_entries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return all_topics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(
            Path(__file__).resolve().parent.parent
            / "knowledge_base"
            / "Materials"
        ),
        help=(
            "Path to the knowledge-base root: either a single "
            "course's chapter folders or a folder containing "
            "multiple course folders."
        ),
    )
    args = parser.parse_args()

    source_root = Path(args.source)
    if not source_root.is_dir():
        raise SystemExit(
            f"Source folder not found: {source_root}"
        )

    with SessionLocal() as session:
        topics = seed_topics(session, source_root)

        print(f"Seeded/updated {len(topics)} topics:")
        for topic in topics:
            print(
                f"  [{topic.topic_id}] "
                f"{topic.topic_key}: {topic.name}"
            )

    print(
        "\nWrote folder->topic map to: "
        f"{TOPIC_FOLDER_MAP_PATH}"
    )


if __name__ == "__main__":
    main()
