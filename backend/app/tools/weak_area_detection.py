import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "weak_area_detection.txt"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Weak area detection prompt file was not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


WEAK_AREA_PROMPT = load_prompt_template()


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


@tool
def detect_weak_areas(assessment_results: str) -> str:
    """
    Identify the learner's weak areas from assessment results.
    """

    prompt = WEAK_AREA_PROMPT.replace(
        "{assessment_results}",
        assessment_results,
    )

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1).strip()

    result = json.loads(content)

    if "weak_areas" not in result:
        raise ValueError(
            "Weak area detection response does not contain 'weak_areas'."
        )

    if not isinstance(result["weak_areas"], list):
        raise ValueError("'weak_areas' must be a list.")

    return json.dumps(result, ensure_ascii=False, indent=2)