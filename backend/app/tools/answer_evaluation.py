import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "answer_evaluation.txt"


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Answer evaluation prompt file was not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


EVALUATION_PROMPT = load_prompt_template()


llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


@tool
def evaluate_answer(
    question: str,
    correct_answer: str,
    learner_answer: str,
) -> str:
    """
    Evaluate a learner's answer and return correctness, score, and feedback.
    """

    prompt = (
        EVALUATION_PROMPT
        .replace("{question}", question)
        .replace("{correct_answer}", correct_answer)
        .replace("{learner_answer}", learner_answer)
    )

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1).strip()

    evaluation = json.loads(content)

    if "is_correct" not in evaluation:
        raise ValueError("Evaluation response does not contain 'is_correct'.")

    if "score_awarded" not in evaluation:
        raise ValueError(
            "Evaluation response does not contain 'score_awarded'."
        )

    if "feedback" not in evaluation:
        raise ValueError("Evaluation response does not contain 'feedback'.")

    if evaluation["is_correct"] not in [True, False]:
        raise ValueError("'is_correct' must be True or False.")

    if evaluation["score_awarded"] not in [0, 1]:
        raise ValueError("'score_awarded' must be 0 or 1.")

    return json.dumps(evaluation, ensure_ascii=False, indent=2)