from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# Load environment variables from .env
load_dotenv()


# Get the project's root directory.
# explain.py is located at:
# MindCompass/backend/app/tools/explain.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROMPT_PATH = PROJECT_ROOT / "prompts" / "explain.txt"


def load_prompt_template() -> str:
    """
    Load the Explain Tool prompt from the prompts directory.

    Returns:
        The prompt template as a string.

    Raises:
        FileNotFoundError:
            If the prompt file does not exist.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Explain prompt file was not found: {PROMPT_PATH}"
        )

    return PROMPT_PATH.read_text(encoding="utf-8")


# Load the prompt once when the module is imported.
EXPLAIN_PROMPT = load_prompt_template()


# Initialize the LLM.
# temperature=0 makes the output more consistent,
# which is useful for a teaching tool.
llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


@tool
def explain(
    topic: str,
    context: str,
    student_level: str,
) -> str:
    """
    Explain a course topic using the provided course context.

    Use this tool when a learner needs an explanation
    of a specific learning topic.

    Args:
        topic:
            The topic that the learner wants to understand.

        context:
            Relevant course content retrieved from the
            knowledge base / RAG system.

        student_level:
            The learner's current level, such as
            beginner, intermediate, or advanced.

    Returns:
        A clear explanation of the requested topic.
    """

    # Insert the actual values into the prompt template.
    prompt = EXPLAIN_PROMPT.format(
        topic=topic,
        context=context,
        student_level=student_level,
    )

    # Send the prompt to the language model.
    response = llm.invoke(prompt)

    # Return only the generated text.
    return response.content