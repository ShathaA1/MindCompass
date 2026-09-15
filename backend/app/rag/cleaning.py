"""Cleans and normalizes content blocks produced by rag/loaders.py.

Applies three passes, in order, to a list of content blocks:

  1. Unicode normalization (NFKC) and control-character stripping.
  2. Removal of repeated header/footer lines (e.g. a running topic
     "kicker" printed on every slide, or a page footer), detected as
     lines that recur identically across most blocks of the same
     source document.
  3. Whitespace collapsing (multiple blank lines / spaces -> single).

Blocks that end up empty after cleaning are dropped.
"""

import re
import unicodedata
from collections import Counter, defaultdict


def normalize_unicode(text: str) -> str:
    """Normalize Unicode form and strip non-printable control characters."""

    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse runs of blank lines, trailing spaces, and repeated spaces."""

    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _line_key(line: str) -> str:
    return line.strip().lower()


def strip_repeated_boilerplate_lines(
    blocks: list[dict], min_ratio: float = 0.5, min_blocks: int = 3
) -> list[dict]:
    """Remove lines that recur identically across most blocks of a document.

    Targets running headers/footers such as a topic "kicker" printed on
    every slide (e.g. a deck about RAG printing the word "RAG" on every
    page), copyright footers, or page numbers -- without touching content
    that is genuinely unique to a given page/cell. Grouping is per
    `source_path` so one document's boilerplate never affects another
    document's blocks.
    """

    by_source: dict[str, list[dict]] = defaultdict(list)
    for block in blocks:
        by_source[block["source_path"]].append(block)

    cleaned_blocks: list[dict] = []
    for source_blocks in by_source.values():
        if len(source_blocks) < min_blocks:
            cleaned_blocks.extend(source_blocks)
            continue

        line_counts: Counter = Counter()
        for block in source_blocks:
            lines_in_block = {
                _line_key(line)
                for line in block["text"].split("\n")
                if line.strip()
            }
            line_counts.update(lines_in_block)

        threshold = max(min_blocks, int(len(source_blocks) * min_ratio))
        boilerplate_keys = {
            key for key, count in line_counts.items() if count >= threshold
        }

        for block in source_blocks:
            kept_lines = [
                line
                for line in block["text"].split("\n")
                if _line_key(line) not in boilerplate_keys
            ]
            new_block = dict(block)
            new_block["text"] = "\n".join(kept_lines).strip()
            cleaned_blocks.append(new_block)

    return cleaned_blocks


def clean_blocks(blocks: list[dict]) -> list[dict]:
    """Run the full cleaning pipeline over a list of loader content blocks."""

    normalized = []
    for block in blocks:
        new_block = dict(block)
        new_block["text"] = normalize_unicode(block["text"])
        normalized.append(new_block)

    deduplicated = strip_repeated_boilerplate_lines(normalized)

    final_blocks = []
    for block in deduplicated:
        text = collapse_whitespace(block["text"])
        if not text:
            continue
        new_block = dict(block)
        new_block["text"] = text
        final_blocks.append(new_block)

    return final_blocks
