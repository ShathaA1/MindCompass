"""Runs the full RAG ingestion pipeline for approved learning materials.

For every lesson folder discovered under --source (Course -> Chapter ->
Lesson -> Files), and for every PDF/ipynb/html file inside it, this
script runs the complete pipeline in order:

    loaders.load_file()        -> extract raw content blocks
    cleaning.clean_blocks()    -> normalize + strip boilerplate
    chunking.chunk_blocks()    -> 250-token chunks, 20% overlap
    embeddings.embed_chunks()  -> OpenAI text-embedding-3-large
    vector_store.upsert_chunks() -> store in ChromaDB with metadata

Each file's progress is tracked in PostgreSQL's `learning_materials`
table via `processing_status` (pending/processing/completed/failed).
Re-running this script SKIPS any file already marked 'completed', so
adding one new PDF to an existing lesson and re-running only processes
that new file -- nothing else is redone.

Prerequisite: scripts/seed_topics.py must have been run first (this
script reads backend/app/rag/topic_folder_map.json, which seed_topics.py
generates, to resolve which topic_id a given lesson folder belongs to).

Usage:
    python -m scripts.ingest_materials --source "/path/to/knowledge_base_root"
    python -m scripts.ingest_materials --source "..." --dry-run
    python -m scripts.ingest_materials --source "..." --no-visuals
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.app.database.connection import SessionLocal
from backend.app.database.models import LearningMaterial
from backend.app.rag.chunking import chunk_blocks
from backend.app.rag.cleaning import clean_blocks
from backend.app.rag.embeddings import embed_chunks
from backend.app.rag.loaders import load_file
from backend.app.rag.vector_store import upsert_chunks
from scripts.seed_topics import TOPIC_FOLDER_MAP_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".ipynb", ".html", ".htm"}


def load_topic_folder_map(path: Path = TOPIC_FOLDER_MAP_PATH) -> dict[str, dict]:
    """Load the folder_path -> topic info map written by seed_topics.py."""

    if not path.exists():
        raise SystemExit(
            f"Topic/folder map not found at {path}. "
            "Run `python -m scripts.seed_topics --source <root>` first."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["folder_path"]: entry for entry in entries}


def discover_files(lesson_dir: Path) -> list[Path]:
    """List every supported source file directly inside a lesson folder."""

    return sorted(
        p
        for p in lesson_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _get_or_create_material(session, topic_id: int, file_path: Path) -> LearningMaterial:
    """Fetch the LearningMaterial row for this file, creating it if needed."""

    material = (
        session.query(LearningMaterial)
        .filter_by(source_path=str(file_path))
        .first()
    )
    if material is None:
        material = LearningMaterial(
            topic_id=topic_id,
            title=file_path.name,
            file_type=file_path.suffix.lstrip(".").lower(),
            source_path=str(file_path),
            processing_status="pending",
            created_at=datetime.now(timezone.utc),
        )
        session.add(material)
        session.flush()
    return material


def ingest_file(
    file_path: Path,
    topic_info: dict,
    describe_visuals: bool,
) -> int:
    """Run loaders -> cleaning -> chunking -> embeddings -> vector_store
    for a single file. Returns the number of chunks stored."""

    blocks = load_file(file_path, describe_visuals=describe_visuals)
    cleaned = clean_blocks(blocks)
    chunks = chunk_blocks(cleaned)
    embedded = embed_chunks(chunks)
    return upsert_chunks(
        embedded,
        topic_id=topic_info["topic_id"],
        course_name=topic_info["course_name"],
        chapter_number=topic_info["chapter_number"],
        lesson_name=topic_info["lesson_name"],
    )


def run_ingestion(
    source_root: Path,
    describe_visuals: bool = True,
    dry_run: bool = False,
) -> dict:
    """Walk every lesson folder and ingest every not-yet-completed file.

    In --dry-run mode, no database writes, embedding calls, or ChromaDB
    writes happen -- it only prints what WOULD be processed, which is
    useful for previewing a run before spending OpenAI credits.
    """

    topic_map = load_topic_folder_map()
    summary = {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

    session = None if dry_run else SessionLocal()
    try:
        for lesson_dir in sorted(
            Path(p) for p in topic_map if Path(p).is_relative_to(source_root)
        ):
            topic_info = topic_map[str(lesson_dir)]
            files = discover_files(lesson_dir)

            for file_path in files:
                if dry_run:
                    logger.info("[DRY RUN] Would ingest: %s (topic_id=%s)", file_path, topic_info["topic_id"])
                    summary["processed"] += 1
                    continue

                material = _get_or_create_material(session, topic_info["topic_id"], file_path)

                if material.processing_status == "completed":
                    logger.info("Skipping already-processed file: %s", file_path)
                    summary["skipped"] += 1
                    continue

                material.processing_status = "processing"
                session.commit()

                try:
                    n_chunks = ingest_file(file_path, topic_info, describe_visuals)
                    material.processing_status = "completed"
                    session.commit()
                    summary["processed"] += 1
                    summary["total_chunks"] += n_chunks
                    logger.info("Ingested %s -> %d chunks", file_path, n_chunks)
                except Exception:
                    session.rollback()
                    material.processing_status = "failed"
                    session.commit()
                    summary["failed"] += 1
                    logger.exception("Failed to ingest %s", file_path)
    finally:
        if session is not None:
            session.close()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Knowledge-base root passed to seed_topics.py (course(s) root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be ingested without calling OpenAI/ChromaDB/Postgres.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip GPT-vision slide descriptions (faster/cheaper, text-only PDFs).",
    )
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source folder not found: {source_root}")

    summary = run_ingestion(
        source_root,
        describe_visuals=not args.no_visuals,
        dry_run=args.dry_run,
    )

    print("\nIngestion summary:")
    print(f"  processed : {summary['processed']}")
    print(f"  skipped   : {summary['skipped']}")
    print(f"  failed    : {summary['failed']}")
    print(f"  chunks    : {summary['total_chunks']}")


if __name__ == "__main__":
    main()
