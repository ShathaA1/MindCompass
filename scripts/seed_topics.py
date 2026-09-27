"""Adds/updates topics and prerequisites in PostgreSQL from the knowledge base.

Walks the knowledge-base source folder in order:

    <source_root>/<course>/<NN_ChapterName>/<NN_NN_LessonName>/files...

i.e. Course -> Chapter -> Lesson -> Files. The number of course folders,
chapters per course, and lessons per chapter are all discovered
dynamically -- nothing is hardcoded, so adding a new course folder
(e.g. "Python_Basics") or a new chapter/lesson just works on the next run.

Idempotent: re-running this script will NOT create duplicate topics.
Each lesson's identity is its exact `name` string (course + chapter +
lesson); if a topic with that name already exists it is reused (and its
learning_objectives/description refreshed) instead of re-inserted.

Also writes `backend/app/rag/topic_folder_map.json`, mapping each lesson
folder's absolute path to its resulting topic_id/course/chapter/lesson --
this is what scripts/ingest_materials.py reads to know which topic_id a
given file belongs to (the `topics` table itself has no folder-path
column, since we are not modifying the given schema).

Usage:
    python -m scripts.seed_topics --source "/path/to/knowledge_base_root"
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Ensure `backend/` is on sys.path so backend/app/database/models.py's
# `from app.database.connection import Base` (a short-form import assuming
# backend/ as root) resolves correctly when this script is run from the
# project root as `python -m scripts.seed_topics`. This is a local,
# non-invasive workaround -- it does not modify any shared file.
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from sqlalchemy.orm import Session

from backend.app.database.connection import SessionLocal
from backend.app.database.models import Topic, TopicPrerequisite

# Temporary uniform value for every seeded topic, per project decision.
# Update manually per-topic later if finer-grained difficulty is needed.
DEFAULT_DIFFICULTY_LEVEL = "intermediate"

TOPIC_FOLDER_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "app"
    / "rag"
    / "topic_folder_map.json"
)


def _clean_title(filename: str) -> str:
    """Turn a slide/lab filename into a readable topic phrase."""

    stem = Path(filename).stem
    stem = re.sub(r"^(slides_|lab_)", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(.*?\)", "", stem)  # drop "(solution)", "(sol)", etc.
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def _title_to_objective(title: str, *, is_practical: bool) -> str:
    """Turn a cleaned title into an actual learning-objective sentence."""

    first_word = title.split(" ", 1)[0] if title else ""
    if first_word.isupper() and len(first_word) > 1:
        lowered = title
    else:
        lowered = title[0].lower() + title[1:] if title else title
    if is_practical:
        return f"Apply {lowered} through hands-on practice"
    return f"Understand {lowered}"


def _extract_learning_objectives(lesson_dir: Path) -> list[str]:
    """Derive learning objectives from slide titles, falling back to labs."""

    slide_files = sorted(lesson_dir.glob("slides_*.pdf"))
    source_files = slide_files or sorted(lesson_dir.glob("lab_*.ipynb"))
    is_practical = not slide_files

    objectives = []
    seen = set()
    for f in source_files:
        title = _clean_title(f.name)
        if not title:
            continue
        objective = _title_to_objective(title, is_practical=is_practical)
        if objective.lower() not in seen:
            objectives.append(objective)
            seen.add(objective.lower())
    return objectives


def _lesson_name_from_folder(folder_name: str) -> str:
    """Strip the numeric prefix (e.g. '01_04_') and tidy remaining underscores."""

    name = re.sub(r"^\d{2}_\d{2}_", "", folder_name).strip()
    name = re.sub(r"[_]+", " ", name)
    return re.sub(r"\s+", " ", name).strip(" :")


def _chapter_number_from_folder(folder_name: str) -> int:
    match = re.match(r"^(\d{2})_", folder_name)
    return int(match.group(1)) if match else 0


def _is_chapter_folder(folder_name: str) -> bool:
    """A chapter folder looks like 'NN_...' (two digits, underscore)."""

    return re.match(r"^\d{2}_", folder_name) is not None


def discover_lessons(source_root: Path) -> list[dict]:
    """Walk course -> chapter -> lesson, in sorted (curriculum) order.

    `source_root` may directly contain chapter folders (single-course
    layout, e.g. the original "Agentic AI" folder) OR contain one or
    more course folders that each contain chapter folders -- both are
    supported without configuration, detected per top-level folder.
    """

    lessons = []
    top_level_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())

    for entry in top_level_dirs:
        if _is_chapter_folder(entry.name):
            # source_root itself is a single course's chapters.
            course_name = source_root.name
            chapter_dirs = [entry]
        elif not any(p.is_dir() for p in entry.iterdir()):
            # Reference-material course: files sit directly in the course
            # folder (no Chapter/Lesson subfolders) -- e.g. a single
            # supplementary textbook used to "fill gaps" rather than a
            # sequenced curriculum. Treat the whole folder as ONE lesson
            # under a synthetic "Chapter 0 - Reference Material".
            course_name = entry.name
            lesson_name = "Reference Material"
            lessons.append(
                {
                    "course_name": course_name,
                    "chapter_number": 0,
                    "lesson_dir": entry,
                    "name": f"{course_name} | Ch0.1 \u2013 {lesson_name}",
                    "description": (
                        f"Course: {course_name} | Supplementary reference "
                        f"material (not part of the sequenced curriculum)."
                    ),
                    "learning_objectives": _extract_learning_objectives(entry),
                }
            )
            continue
        else:
            # entry is a course folder; its children are chapters.
            course_name = entry.name
            chapter_dirs = sorted(p for p in entry.iterdir() if p.is_dir())

        for chapter_dir in chapter_dirs:
            chapter_number = _chapter_number_from_folder(chapter_dir.name)
            lesson_dirs = sorted(p for p in chapter_dir.iterdir() if p.is_dir())

            if not lesson_dirs:
                # Chapter has files directly inside it (no Lesson
                # subfolders) -- e.g. a reference book split into one
                # PDF per chapter. Treat the whole chapter as ONE lesson.
                chapter_title = re.sub(
                    r"^\d{2}_(Chapter_\d+_)?", "", chapter_dir.name
                )
                chapter_title = re.sub(r"[_]+", " ", chapter_title).strip()
                lessons.append(
                    {
                        "course_name": course_name,
                        "chapter_number": chapter_number,
                        "lesson_dir": chapter_dir,
                        "name": (
                            f"{course_name} | Ch{chapter_number}.1 "
                            f"\u2013 {chapter_title}"
                        ),
                        "description": (
                            f"Course: {course_name} | Chapter "
                            f"{chapter_number} lesson: {chapter_title}"
                        ),
                        "learning_objectives": _extract_learning_objectives(
                            chapter_dir
                        ),
                    }
                )
                continue

            for lesson_dir in lesson_dirs:
                lesson_name = _lesson_name_from_folder(lesson_dir.name)
                lesson_number = int(lesson_dir.name.split("_")[1])
                lessons.append(
                    {
                        "course_name": course_name,
                        "chapter_number": chapter_number,
                        "lesson_dir": lesson_dir,
                        "name": (
                            f"{course_name} | Ch{chapter_number}.{lesson_number} "
                            f"\u2013 {lesson_name}"
                        ),
                        "description": (
                            f"Course: {course_name} | Chapter {chapter_number} "
                            f"lesson: {lesson_name}"
                        ),
                        "learning_objectives": _extract_learning_objectives(
                            lesson_dir
                        ),
                    }
                )

    # Deduplicate courses that legitimately appear twice on disk (e.g. a
    # course folder was accidentally nested inside itself) is NOT handled
    # here on purpose -- discovery reflects exactly what's on disk.
    return lessons


def seed_topics(session: Session, source_root: Path) -> list[Topic]:
    """Insert or update one Topic per lesson; idempotent across re-runs.

    Chains each lesson to the previous lesson *in the same course AND
    chapter* via `topic_prerequisites` (so chapter "1" in one course
    doesn't get chained to chapter "1" of a different course).
    """

    lessons = discover_lessons(source_root)

    existing_topics_by_name = {t.name: t for t in session.query(Topic).all()}

    all_topics: list[Topic] = []
    folder_map_entries: list[dict] = []
    previous_topic_by_course_chapter: dict[tuple[str, int], Topic] = {}

    for lesson in lessons:
        topic = existing_topics_by_name.get(lesson["name"])
        if topic is None:
            topic = Topic(
                name=lesson["name"],
                description=lesson["description"],
                difficulty_level=DEFAULT_DIFFICULTY_LEVEL,
                learning_objectives=lesson["learning_objectives"],
            )
            session.add(topic)
            session.flush()  # assign topic.topic_id before referencing it
            existing_topics_by_name[lesson["name"]] = topic
        else:
            # Refresh derived fields in case source files changed.
            topic.description = lesson["description"]
            topic.learning_objectives = lesson["learning_objectives"]

        key = (lesson["course_name"], lesson["chapter_number"])
        previous_topic = previous_topic_by_course_chapter.get(key)
        if previous_topic is not None:
            already_linked = (
                session.query(TopicPrerequisite)
                .filter_by(
                    topic_id=topic.topic_id,
                    prerequisite_topic_id=previous_topic.topic_id,
                )
                .first()
            )
            if already_linked is None:
                session.add(
                    TopicPrerequisite(
                        topic_id=topic.topic_id,
                        prerequisite_topic_id=previous_topic.topic_id,
                    )
                )
        previous_topic_by_course_chapter[key] = topic

        all_topics.append(topic)
        folder_map_entries.append(
            {
                "folder_path": str(lesson["lesson_dir"].resolve()),
                "topic_id": topic.topic_id,
                "course_name": lesson["course_name"],
                "chapter_number": lesson["chapter_number"],
                "lesson_name": lesson["name"],
            }
        )

    session.commit()

    TOPIC_FOLDER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_FOLDER_MAP_PATH.write_text(
        json.dumps(folder_map_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return all_topics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parent.parent / "knowledge_base" / "Materials"),
        help=(
            "Path to the knowledge-base root: either a single course's "
            "chapter folders directly, or a folder containing one or "
            "more course folders. Defaults to knowledge_base/Materials "
            "at the project root."
        ),
    )
    args = parser.parse_args()

    source_root = Path(args.source)
    if not source_root.is_dir():
        raise SystemExit(f"Source folder not found: {source_root}")

    with SessionLocal() as session:
        topics = seed_topics(session, source_root)

        print(f"Seeded/updated {len(topics)} topics:")
        for topic in topics:
            print(f"  [{topic.topic_id}] {topic.name}")

    print(f"\nWrote folder->topic_id map to: {TOPIC_FOLDER_MAP_PATH}")


if __name__ == "__main__":
    main()
