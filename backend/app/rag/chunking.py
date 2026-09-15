"""Splits cleaned content blocks into overlapping, token-sized chunks.

Design decisions locked in for this project:
  - chunk size: 250 tokens
  - overlap: 20% (50 tokens)
  - consecutive blocks (pages/cells) from the SAME source file are
    merged into one continuous token stream before splitting, so a
    chunk is not forced to stop at a page/cell boundary just because
    the page/cell itself is shorter than 250 tokens.

Tokenization uses tiktoken's `cl100k_base` encoding, which is the
encoding used by the `text-embedding-3-large` embedding model.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import tiktoken

CHUNK_SIZE_TOKENS = 250
OVERLAP_RATIO = 0.20
OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * OVERLAP_RATIO)  # 50 tokens

_encoding = None


def _get_encoding():
    """Lazily load the tiktoken encoding (downloaded once, then cached)."""

    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


@dataclass
class TokenSpan:
    """One block's tokens, tagged with the unit_index they came from."""

    unit_index: int
    unit_type: str
    tokens: list[int] = field(default_factory=list)


def _build_token_stream(blocks: list[dict]) -> tuple[list[int], list[int], list[str]]:
    """Flatten a document's blocks into one token stream.

    Returns (tokens, unit_index_per_token, unit_type_per_token) -- three
    parallel lists so that, after slicing a window of tokens, we can look
    up which original page(s)/cell(s) it came from.
    """

    encoding = _get_encoding()
    tokens: list[int] = []
    unit_index_per_token: list[int] = []
    unit_type_per_token: list[str] = []

    for block in blocks:
        block_tokens = encoding.encode(block["text"])
        # A blank separator token stream between blocks keeps chunk text
        # readable; we tag the separator with the block it follows.
        if tokens:
            sep_tokens = encoding.encode("\n\n")
            tokens.extend(sep_tokens)
            unit_index_per_token.extend([block["unit_index"]] * len(sep_tokens))
            unit_type_per_token.extend([block["unit_type"]] * len(sep_tokens))

        tokens.extend(block_tokens)
        unit_index_per_token.extend([block["unit_index"]] * len(block_tokens))
        unit_type_per_token.extend([block["unit_type"]] * len(block_tokens))

    return tokens, unit_index_per_token, unit_type_per_token


def chunk_document_blocks(
    blocks: list[dict],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Chunk all blocks belonging to ONE source document.

    Consecutive blocks are merged into a single token stream first, so
    chunks can span multiple pages/cells to reach the target size.
    """

    if not blocks:
        return []

    encoding = _get_encoding()
    blocks = sorted(blocks, key=lambda b: b["unit_index"])
    tokens, unit_index_per_token, unit_type_per_token = _build_token_stream(blocks)

    stride = chunk_size - overlap_tokens
    if stride <= 0:
        raise ValueError("overlap_tokens must be smaller than chunk_size")

    source_path = blocks[0]["source_path"]
    file_type = blocks[0]["file_type"]

    chunks: list[dict] = []
    chunk_index = 0
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        window_tokens = tokens[start:end]
        window_units = unit_index_per_token[start:end]
        window_unit_types = set(unit_type_per_token[start:end])

        chunk_text = encoding.decode(window_tokens).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "source_path": source_path,
                    "file_type": file_type,
                    "unit_type": (
                        window_unit_types.pop()
                        if len(window_unit_types) == 1
                        else "mixed"
                    ),
                    "start_unit_index": min(window_units),
                    "end_unit_index": max(window_units),
                    "chunk_index": chunk_index,
                    "token_count": len(window_tokens),
                }
            )
            chunk_index += 1

        if end == total_tokens:
            break
        start += stride

    return chunks


def chunk_blocks(
    blocks: list[dict],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Chunk blocks that may come from MULTIPLE source documents.

    Groups by `source_path` first (so overlap/merging never crosses a
    document boundary), then chunks each document independently.
    """

    by_source: dict[str, list[dict]] = defaultdict(list)
    for block in blocks:
        by_source[block["source_path"]].append(block)

    all_chunks: list[dict] = []
    for source_blocks in by_source.values():
        all_chunks.extend(
            chunk_document_blocks(
                source_blocks, chunk_size=chunk_size, overlap_tokens=overlap_tokens
            )
        )
    return all_chunks
