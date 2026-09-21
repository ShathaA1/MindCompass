"""Stores assessments, processes responses, and updates learner results."""

import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.database.models import (
    AssessmentAttempt,
    AssessmentResponse,
    TopicMastery,
)
from app.tools.weak_area_detection import detect_weak_areas


def _calculate_topic_scores(
    questions: list[dict],
) -> dict[int, tuple[int, int]]:
    """Calculate correct answers and total questions for each topic."""

    topic_scores: dict[int, tuple[int, int]] = {}

    for question in questions:
        topic_id = question.get("topic_id")

        if topic_id is None:
            continue

        correct, total = topic_scores.get(topic_id, (0, 0))

        total += 1

        if question.get("is_correct") is True:
            correct += 1

        topic_scores[topic_id] = (correct, total)

    return topic_scores


def _detect_topic_weak_areas(
    topic_id: int,
    questions: list[dict],
) -> list:
    """Detect weak areas for a specific topic."""

    topic_questions = [
        question
        for question in questions
        if question.get("topic_id") == topic_id
    ]

    assessment_results = json.dumps(
        topic_questions,
        ensure_ascii=False,
        indent=2,
    )

    result = detect_weak_areas.invoke(
        {
            "assessment_results": assessment_results,
        }
    )

    parsed_result = json.loads(result)

    return parsed_result.get("weak_areas", [])


def _update_topic_mastery(
    db: Session,
    user_id: int,
    topic_id: int,
    mastery_score: float,
    weak_areas: list,
) -> TopicMastery:
    """Create or update the learner's mastery record for a topic."""

    mastery = (
        db.query(TopicMastery)
        .filter(
            TopicMastery.user_id == user_id,
            TopicMastery.topic_id == topic_id,
        )
        .first()
    )

    now = datetime.now(timezone.utc)

    if mastery is None:
        mastery = TopicMastery(
            user_id=user_id,
            topic_id=topic_id,
            mastery_score=mastery_score,
            weak_areas=weak_areas,
            last_assessed_at=now,
        )

        db.add(mastery)

    else:
        mastery.mastery_score = mastery_score
        mastery.weak_areas = weak_areas
        mastery.last_assessed_at = now

    return mastery


def save_assessment_result(
    db: Session,
    user_id: int,
    topic_id: int | None,
    assessment_type: str,
    questions: list[dict],
    feedback: str | None = None,
) -> AssessmentAttempt:
    """
    Save an assessment attempt and all of its responses.

    The function also calculates topic-level mastery and weak areas.
    """

    if not questions:
        raise ValueError("At least one question is required.")

    total_questions = len(questions)

    correct_answers = sum(
        1
        for question in questions
        if question.get("is_correct") is True
    )

    now = datetime.now(timezone.utc)

    attempt = AssessmentAttempt(
        user_id=user_id,
        topic_id=topic_id,
        assessment_type=assessment_type,
        score=correct_answers,
        max_score=total_questions,
        feedback=feedback,
        started_at=now,
        completed_at=now,
    )

    db.add(attempt)
    db.flush()

    for question in questions:
        response = AssessmentResponse(
            attempt_id=attempt.assessment_attempt_id,
            topic_id=question["topic_id"],
            question_text=question["question_text"],
            question_type=question["question_type"],
            options=question.get("options"),
            learner_answer=question.get("learner_answer"),
            correct_answer=question.get("correct_answer"),
            is_correct=question.get("is_correct"),
            score_awarded=question.get("score_awarded"),
            feedback=question.get("feedback"),
        )

        db.add(response)

    topic_scores = _calculate_topic_scores(questions)

    for current_topic_id, (correct, total) in topic_scores.items():
        mastery_score = (correct / total) * 100

        try:
            weak_areas = _detect_topic_weak_areas(
                current_topic_id,
                questions,
            )
        except (ValueError, json.JSONDecodeError):
            weak_areas = []

        _update_topic_mastery(
            db=db,
            user_id=user_id,
            topic_id=current_topic_id,
            mastery_score=mastery_score,
            weak_areas=weak_areas,
        )

    db.commit()
    db.refresh(attempt)

    return attempt