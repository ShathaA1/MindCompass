from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load environment variables from the project .env file.
load_dotenv()


# Initialize a deterministic model for intent classification.
intent_llm = ChatOpenAI(
    model="gpt-5.6-luna",
    temperature=0,
)


VALID_INTENTS = {
    "explain",
    "practice",
    "assess",
    "review",
    "recommend",
}


def classify_learner_intent(
    user_message: str,
    conversation_history: list | None = None,
) -> str:
    """
    Classify the learner's current intent using
    both the latest message and recent conversation context.
    """

    conversation_history = conversation_history or []

    # Keep only a small recent window because the classifier
    # only needs enough context to understand follow-up messages.
    recent_messages = conversation_history[-6:]

    history_text = "\n".join(
        (
            f"{message.get('role', 'unknown')}: "
            f"{message.get('content', '')}"
        )
        for message in recent_messages
    )

    prompt = f"""
You are an intent classifier for an AI tutoring system.

Classify the learner's latest message into exactly ONE
of these intents:

- explain
- practice
- assess
- review
- recommend

Intent definitions:

explain:
The learner is asking for information, clarification,
examples, applications, comparisons, definitions,
simplification, follow-up explanation, or asking a
question about the topic.

practice:
The learner explicitly wants exercises, practice,
flashcards, coding practice, scenarios, or another
practice activity.

assess:
The learner explicitly wants to be tested, assessed,
quizzed, or wants their understanding evaluated.

review:
The learner explicitly wants to review, revise,
recap, summarize, or revisit weak areas.

recommend:
The learner asks what to do next, wants to continue,
or does not make another clear learning request.

Important rules:

1. A normal learner question should be classified
   as explain.

2. Follow-up questions must use the conversation
   history to understand their meaning.

3. Messages such as:
   "what about finance?"
   "and banking?"
   "more examples?"
   "tell me more"
   are explain when they continue a previous
   explanation.

4. Do not classify examples as practice unless
   the learner explicitly asks for a practice
   activity or exercise.

5. Return ONLY the intent name.
   Do not include an explanation.

Recent conversation:
{history_text}

Latest learner message:
{user_message}
"""

    try:
        response = intent_llm.invoke(prompt)

        intent = response.content.strip().lower()

        if intent in VALID_INTENTS:
            return intent

    except Exception:
        # Fall back safely when the classifier
        # cannot reach the language model.
        pass

    return "recommend"