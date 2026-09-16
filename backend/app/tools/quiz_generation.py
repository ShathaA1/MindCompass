import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "quiz_generation.txt"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Quiz generation prompt file was not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


QUIZ_PROMPT = load_prompt_template()


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


@tool
def generate_quiz(
    topic: str,
    context: str,
    student_level: str,
    num_questions: int = 5,
    assessment_type: str = "placement",
) -> str:
    """
    Generate a multiple-choice quiz based only on the provided learning context.
    """

    prompt = (
        QUIZ_PROMPT
        .replace("{topic}", topic)
        .replace("{context}", context)
        .replace("{student_level}", student_level)
        .replace("{num_questions}", str(num_questions))
        .replace("{assessment_type}", assessment_type))

    response = llm.invoke(prompt)

    content = response.content.strip()

    # Remove Markdown code fences if the model adds them
    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1).strip()

    # Validate that the response is valid JSON
    quiz = json.loads(content)

    # Validate the main structure
    if "questions" not in quiz:
        raise ValueError("Quiz response does not contain 'questions'.")

    if len(quiz["questions"]) != num_questions:
        raise ValueError(
            f"Expected {num_questions} questions, "
            f"but received {len(quiz['questions'])}."
        )

    return json.dumps(quiz, ensure_ascii=False, indent=2)