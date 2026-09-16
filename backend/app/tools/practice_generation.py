import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "practice_generation.txt"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Practice generation prompt file was not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


PRACTICE_PROMPT = load_prompt_template()


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


@tool
def generate_practice(
    topic: str,
    context: str,
    student_level: str,
    practice_type: str = "flashcards",
    num_items: int = 5,
    weak_areas: str = "[]",
) -> str:
    """
    Generate a practice activity based on the provided learning context.
    """

    prompt = (
        PRACTICE_PROMPT
        .replace("{topic}", topic)
        .replace("{context}", context)
        .replace("{student_level}", student_level)
        .replace("{practice_type}", practice_type)
        .replace("{num_items}", str(num_items))
        .replace("{weak_areas}", weak_areas)
    )

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1).strip()

    practice = json.loads(content)

    if "practice_type" not in practice:
        raise ValueError(
            "Practice response does not contain 'practice_type'."
        )

    if "items" not in practice:
        raise ValueError(
            "Practice response does not contain 'items'."
        )

    if practice["practice_type"] != practice_type:
        raise ValueError(
            f"Expected practice type '{practice_type}', "
            f"but received '{practice['practice_type']}'."
        )

    if not isinstance(practice["items"], list):
        raise ValueError("'items' must be a list.")

    if len(practice["items"]) != num_items:
        raise ValueError(
            f"Expected {num_items} items, "
            f"but received {len(practice['items'])}."
        )

    for item in practice["items"]:
        if "prompt" not in item:
            raise ValueError("Each practice item must contain 'prompt'.")

        if "answer" not in item:
            raise ValueError("Each practice item must contain 'answer'.")

    return json.dumps(practice, ensure_ascii=False, indent=2)