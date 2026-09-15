"""Evaluates RAG retrieval and answer quality using an LLM-as-judge.

For each test question:
  1. Retrieve top_k chunks (optionally scoped to one or more topic_ids).
  2. Generate an answer with the chat model, instructed to answer ONLY
     from the retrieved context (hallucination control).
  3. Ask the chat model to judge, on a 1-5 scale, the three metrics
     from the project brief:
       - Context Relevance: how relevant the retrieved chunks are to the question
       - Groundedness / Faithfulness: whether the answer sticks to the context
       - Answer Relevance: whether the answer actually addresses the question

This complements (not replaces) manual review: raw retrieved chunks are
always included in the result so a human can also read them directly.
"""

import json
from dataclasses import dataclass, field

from openai import OpenAI

from backend.app.core.config import get_settings
from backend.app.rag.retrieval import retrieve

settings = get_settings()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


ANSWER_SYSTEM_PROMPT = (
    "You are a study assistant. Answer the learner's question using ONLY "
    "the provided context. If the context does not contain the answer, "
    "say you don't have enough information in the material -- do not use "
    "outside knowledge and do not guess."
)

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating a Retrieval-Augmented Generation (RAG) system. "
    "Score the following on an integer 1-5 scale each:\n"
    "- context_relevance: how relevant the retrieved context is to the question\n"
    "- groundedness: whether the answer is fully supported by the context "
    "(no facts stated that are not present in the context)\n"
    "- answer_relevance: whether the answer directly and completely "
    "addresses the question\n"
    "Respond with ONLY a JSON object of the exact shape: "
    '{"context_relevance": int, "groundedness": int, '
    '"answer_relevance": int, "notes": str}'
)


@dataclass
class EvaluationResult:
    """The full record for one evaluated question, for review or logging."""

    question: str
    topic_ids: int | list[int] | None
    retrieved_chunks: list[dict] = field(default_factory=list)
    answer: str = ""
    context_relevance: int = 0
    groundedness: int = 0
    answer_relevance: int = 0
    notes: str = ""

    @property
    def average_score(self) -> float:
        return (
            self.context_relevance + self.groundedness + self.answer_relevance
        ) / 3


def _join_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(c["text"] for c in chunks)


def generate_answer(question: str, chunks: list[dict], model: str | None = None) -> str:
    """Answer a question strictly from the retrieved chunks."""

    client = _get_client()
    response = client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{_join_context(chunks)}\n\nQuestion: {question}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def judge_answer(
    question: str, chunks: list[dict], answer: str, model: str | None = None
) -> dict:
    """Score context_relevance / groundedness / answer_relevance via LLM judge."""

    client = _get_client()
    response = client.chat.completions.create(
        model=model or settings.openai_chat_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Context:\n{_join_context(chunks)}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def evaluate_question(
    question: str,
    topic_ids: int | list[int] | None = None,
    top_k: int = 5,
) -> EvaluationResult:
    """Run retrieval + answer generation + LLM-judge scoring for one question."""

    chunks = retrieve(question, topic_ids=topic_ids, top_k=top_k)
    if not chunks:
        return EvaluationResult(
            question=question,
            topic_ids=topic_ids,
            retrieved_chunks=[],
            answer="",
            notes="No chunks retrieved -- check topic_id filter and ingestion.",
        )

    answer = generate_answer(question, chunks)
    scores = judge_answer(question, chunks, answer)

    return EvaluationResult(
        question=question,
        topic_ids=topic_ids,
        retrieved_chunks=chunks,
        answer=answer,
        context_relevance=int(scores.get("context_relevance", 0)),
        groundedness=int(scores.get("groundedness", 0)),
        answer_relevance=int(scores.get("answer_relevance", 0)),
        notes=scores.get("notes", ""),
    )


def run_evaluation_suite(test_cases: list[dict]) -> list[EvaluationResult]:
    """Run evaluate_question over a list of test cases.

    Each test case is a dict: {"question": str, "topic_ids": int|list|None,
    "top_k": int (optional)}.
    """

    return [
        evaluate_question(
            case["question"],
            topic_ids=case.get("topic_ids"),
            top_k=case.get("top_k", 5),
        )
        for case in test_cases
    ]
