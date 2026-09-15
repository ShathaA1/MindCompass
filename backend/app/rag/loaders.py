"""Extracts educational content from PDF, Jupyter Notebook, and HTML files.

Each loader returns a list of "content blocks" -- one dict per logical
unit (a PDF page, a notebook cell, or a whole HTML page) -- with a
consistent shape so downstream cleaning/chunking (Task 3/4) can treat
every source type the same way:

    {
        "text": str,
        "source_path": str,
        "file_type": "pdf" | "ipynb" | "html",
        "unit_type": str,   # e.g. "slide_page", "cell_markdown", "cell_code"
        "unit_index": int,  # page number / cell index / 0 for whole-page HTML
    }

PDF slides in this knowledge base are often mostly images (screenshots,
diagrams, memes) with very little selectable text, so `extract_pdf_content`
combines the page's raw text with a short GPT-vision description of any
diagrams/equations/tables/charts on the page.
"""

import base64
from pathlib import Path

import nbformat
import pymupdf
from bs4 import BeautifulSoup
from openai import OpenAI

from backend.app.core.config import get_settings

settings = get_settings()
_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    """Lazily create a single shared OpenAI client."""

    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


VISUAL_DESCRIPTION_PROMPT = (
    "You are helping build a study knowledge base from a lecture slide. "
    "Look at this slide image and describe ONLY the meaningful visual "
    "content that is not already plain text: diagrams, charts, flowcharts, "
    "tables, architecture drawings, or mathematical equations. Summarize "
    "what they show in 1-3 concise sentences. If the slide has no "
    "diagrams/charts/tables/equations (e.g. it's just text, a logo, a "
    "decorative image, or a meme), respond with exactly: NONE."
)


def _describe_page_visuals(image_bytes: bytes, model: str | None = None) -> str:
    """Ask GPT vision to describe diagrams/equations/tables on a rendered page.

    Returns an empty string when the model reports no meaningful visual
    content (e.g. decorative images), so callers can skip appending it.
    """

    client = _get_openai_client()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISUAL_DESCRIPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
        max_tokens=200,
    )
    description = (response.choices[0].message.content or "").strip()
    return "" if description.upper() == "NONE" else description


def extract_pdf_content(
    path: Path, describe_visuals: bool = True, render_dpi: int = 150
) -> list[dict]:
    """Extract per-page text (+ optional GPT vision descriptions) from a PDF."""

    doc = pymupdf.open(path)
    blocks: list[dict] = []
    try:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            text = page.get_text().strip()

            visual_description = ""
            if describe_visuals:
                pixmap = page.get_pixmap(dpi=render_dpi)
                visual_description = _describe_page_visuals(pixmap.tobytes("png"))

            combined_text = text
            if visual_description:
                combined_text = (
                    f"{text}\n\n[Visual content: {visual_description}]".strip()
                )

            if not combined_text:
                continue

            blocks.append(
                {
                    "text": combined_text,
                    "source_path": str(path),
                    "file_type": "pdf",
                    "unit_type": "slide_page",
                    "unit_index": page_number,
                }
            )
    finally:
        doc.close()
    return blocks


def _extract_output_text(output: dict) -> str:
    """Pull readable text out of a single notebook cell output."""

    output_type = output.get("output_type")
    if output_type == "stream":
        return "".join(output.get("text", [])).strip()
    if output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        text = data.get("text/plain")
        if text is not None:
            return ("".join(text) if isinstance(text, list) else str(text)).strip()
    if output_type == "error":
        return "\n".join(output.get("traceback", [])).strip()
    return ""


def extract_notebook_content(path: Path) -> list[dict]:
    """Extract markdown + code + text outputs from a Jupyter notebook.

    Includes markdown cells, code cells, and each code cell's text
    outputs (stdout/stream text, execute_result/display_data text/plain,
    and error tracebacks), as decided for this project.
    """

    notebook = nbformat.read(path, as_version=4)
    blocks: list[dict] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            text = cell.source.strip()
            unit_type = "cell_markdown"

        elif cell.cell_type == "code":
            parts = []
            code = cell.source.strip()
            if code:
                parts.append(f"```python\n{code}\n```")
            for output in cell.get("outputs", []):
                output_text = _extract_output_text(output)
                if output_text:
                    parts.append(f"Output:\n{output_text}")
            text = "\n\n".join(parts).strip()
            unit_type = "cell_code"

        else:
            continue

        if not text:
            continue

        blocks.append(
            {
                "text": text,
                "source_path": str(path),
                "file_type": "ipynb",
                "unit_type": unit_type,
                "unit_index": cell_index,
            }
        )

    return blocks


def extract_html_content(path: Path) -> list[dict]:
    """Extract the main readable text from a saved HTML page."""

    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    raw_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    body_text = "\n".join(lines)

    if not body_text:
        return []

    full_text = f"{title}\n\n{body_text}" if title else body_text
    return [
        {
            "text": full_text,
            "source_path": str(path),
            "file_type": "html",
            "unit_type": "html_page",
            "unit_index": 0,
        }
    ]


def load_file(path: Path, describe_visuals: bool = True) -> list[dict]:
    """Dispatch to the correct loader based on file extension."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_content(path, describe_visuals=describe_visuals)
    if suffix == ".ipynb":
        return extract_notebook_content(path)
    if suffix in (".html", ".htm"):
        return extract_html_content(path)
    raise ValueError(f"Unsupported file type: {path}")
