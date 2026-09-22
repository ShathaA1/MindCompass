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
    topics: list[dict],
    context: str,
    student_level: str,
    num_questions: int = 8,
    assessment_type: str = "diagnostic",
) -> str:
    """
    Generate a multiple-choice quiz based only on the provided learning context.

    Each question is assigned to one of the provided topics using its topic_id.
    """

    if not topics:
        raise ValueError("At least one topic is required.")

    if num_questions < 1:
        raise ValueError("num_questions must be at least 1.")

    # Validate topic structure
    for topic in topics:
        if "topic_id" not in topic or "topic" not in topic:
            raise ValueError(
                "Each topic must contain 'topic_id' and 'topic'."
            )

    topics_text = "\n".join(
        f"- topic_id: {topic['topic_id']}, topic: {topic['topic']}"
        for topic in topics
    )

    prompt = (
        QUIZ_PROMPT
        .replace("{topics}", topics_text)
        .replace("{context}", context)
        .replace("{student_level}", student_level)
        .replace("{num_questions}", str(num_questions))
        .replace("{assessment_type}", assessment_type)
    )

    response = llm.invoke(prompt)

    content = response.content.strip()

    # Remove Markdown code fences if the model adds them
    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "", 1).strip()

    # Validate that the response is valid JSON
    try:
        quiz = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Quiz response is not valid JSON.") from exc

    # Validate the main structure
    if "questions" not in quiz:
        print("\n--- RAW QUIZ RESPONSE ---")
        print(content)
        print("\n--- PARSED JSON ---")
        print(quiz)
        print("-------------------------\n")

        raise ValueError("Quiz response does not contain 'questions'.")

    if len(quiz["questions"]) != num_questions:
        raise ValueError(
            f"Expected {num_questions} questions, "
            f"but received {len(quiz['questions'])}."
        )

    # Get valid topic IDs
    valid_topic_ids = {
        topic["topic_id"]
        for topic in topics
    }

    # Validate every question
    for question in quiz["questions"]:
        if "topic_id" not in question:
            raise ValueError(
                "A quiz question is missing 'topic_id'."
            )

        if question["topic_id"] not in valid_topic_ids:
            raise ValueError(
                f"Invalid topic_id: {question['topic_id']}"
            )

        required_fields = [
            "question_text",
            "question_type",
            "difficulty",
            "options",
            "correct_answer",
        ]

        for field in required_fields:
            if field not in question:
                raise ValueError(
                    f"Quiz question is missing '{field}'."
                )

    return json.dumps(
        quiz,
        ensure_ascii=False,
        indent=2
    )