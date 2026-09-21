import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "feedback_generation.txt"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Feedback generation prompt file was not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


FEEDBACK_PROMPT = load_prompt_template()


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


@tool
def generate_feedback(
    assessment_results: str,
    weak_areas: str,
    student_level: str,
) -> str:
    """
    Generate personalized feedback from assessment results and weak areas.
    """

    prompt = (
        FEEDBACK_PROMPT
        .replace("{assessment_results}", assessment_results)
        .replace("{weak_areas}", weak_areas)
        .replace("{student_level}", student_level)
    )

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1).strip()

    feedback = json.loads(content)

    required_fields = [
        "summary",
        "strengths",
        "weak_areas",
        "recommendation",
    ]

    for field in required_fields:
        if field not in feedback:
            raise ValueError(
                f"Feedback response does not contain '{field}'."
            )

    if not isinstance(feedback["strengths"], list):
        raise ValueError("'strengths' must be a list.")

    if not isinstance(feedback["weak_areas"], list):
        raise ValueError("'weak_areas' must be a list.")

    return json.dumps(feedback, ensure_ascii=False, indent=2)