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


def classify_topic_scope(
    user_message: str,
    current_topic_name: str,
) -> str:
    """
    Determine whether the learner's request belongs
    to the current learning topic.

    Returns:
        - in_scope
        - no_explicit_topic
        - out_of_scope
    """

    # Treat empty or generic requests as referring
    # to the learner's current topic.
    if not user_message.strip():
        return "no_explicit_topic"

    prompt = f"""
You are a topic-scope classifier for an AI tutoring system.

Current learning topic:
{current_topic_name}

Learner request:
{user_message}

Classify the learner request into exactly one category:

in_scope
- The request is clearly about the current topic.
- The learner may ask about a concept, subtopic, example,
  application, clarification, practice activity, or deeper
  explanation related to the current topic.
- The exact current topic name does not need to appear.

no_explicit_topic
- The learner asks for an activity or continuation without
  naming a different subject.
- Examples:
  "Explain more"
  "Give me 3 flashcards"
  "Give me another example"
  "Quiz me"
  "Let's practice"
  "Can you explain that again?"

out_of_scope
- The learner explicitly asks to learn, explain, practice,
  review, or be assessed on a different topic.
- This includes another course topic even if it exists
  elsewhere in the learning path.

Examples:

Current topic: Prompt Engineering
Request: "Explain zero-shot prompting"
Answer: in_scope

Current topic: Prompt Engineering
Request: "Give me 2 scenarios about prompt optimization"
Answer: in_scope

Current topic: Prompt Engineering
Request: "Give me 3 flashcards"
Answer: no_explicit_topic

Current topic: Prompt Engineering
Request: "Explain more"
Answer: no_explicit_topic

Current topic: Prompt Engineering
Request: "Give me 3 flashcards about RAG"
Answer: out_of_scope

Current topic: Prompt Engineering
Request: "Quiz me on transformers"
Answer: out_of_scope

Current topic: Prompt Engineering
Request: "Explain Python functions"
Answer: out_of_scope

Return only one of:
in_scope
no_explicit_topic
out_of_scope
"""

    try:
        response = intent_llm.invoke(prompt)

        result = response.content.strip().lower()

        allowed_results = {
            "in_scope",
            "no_explicit_topic",
            "out_of_scope",
        }

        if result in allowed_results:
            return result

    except Exception:
        pass

    # Use a permissive fallback so temporary classifier
    # failures do not block normal tutoring.
    return "no_explicit_topic"