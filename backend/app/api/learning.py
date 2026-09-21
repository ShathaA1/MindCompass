from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db
from app.database.models import (
    LearnerProfile,
    LearningPath,
    LearningPathItem,
    Topic,
    TopicMastery,
)

from app.agent.graph import build_tutor_graph
from app.schemas.assessment import (
    DiagnosticStartRequest,
    DiagnosticSubmitRequest,
)


router = APIRouter(prefix="/learning", tags=["Learning"])


@router.get("/dashboard")
def dashboard(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = int(current_user["sub"])

    profile = (
        db.query(LearnerProfile)
        .filter(LearnerProfile.user_id == user_id)
        .first()
    )

    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Learner profile not found",
        )

    learning_path = (
        db.query(LearningPath)
        .filter(LearningPath.user_id == user_id)
        .order_by(LearningPath.created_at.desc())
        .first()
    )

    masteries = (
        db.query(TopicMastery)
        .filter(TopicMastery.user_id == user_id)
        .all()
    )

    return {
        "goal": profile.goal,
        "current_level": profile.current_level,
        "weekly_hours": profile.weekly_hours,
        "learning_path": (
            learning_path.name if learning_path else None
        ),
        "topics_assessed": len(masteries),
    }


@router.get("/progress")
def progress(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = int(current_user["sub"])

    results = (
        db.query(TopicMastery, Topic)
        .join(
            Topic,
            TopicMastery.topic_id == Topic.topic_id,
        )
        .filter(TopicMastery.user_id == user_id)
        .all()
    )

    return [
        {
            "topic_id": topic.topic_id,
            "topic_name": topic.name,
            "mastery_score": mastery.mastery_score,
            "weak_areas": mastery.weak_areas,
            "last_assessed_at": mastery.last_assessed_at,
        }
        for mastery, topic in results
    ]


@router.get("/learning-path")
def learning_path(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = int(current_user["sub"])

    path = (
        db.query(LearningPath)
        .filter(LearningPath.user_id == user_id)
        .order_by(LearningPath.created_at.desc())
        .first()
    )

    if not path:
        return {
            "learning_path": None,
            "items": [],
        }

    items = (
        db.query(LearningPathItem, Topic)
        .join(
            Topic,
            LearningPathItem.topic_id == Topic.topic_id,
        )
        .filter(
            LearningPathItem.learning_path_id
            == path.learning_path_id
        )
        .order_by(LearningPathItem.position)
        .all()
    )

    return {
        "learning_path": {
            "learning_path_id": path.learning_path_id,
            "name": path.name,
            "goal": path.goal,
            "status": path.status,
        },
        "items": [
            {
                "position": item.position,
                "topic_id": topic.topic_id,
                "topic_name": topic.name,
                "status": item.status,
                "recommended_action": item.recommended_action,
            }
            for item, topic in items
        ],
    }


@router.post("/diagnostic/start")
def start_initial_diagnostic(
    data: DiagnosticStartRequest,
    current_user=Depends(get_current_user),
):
    """
    Start the initial diagnostic for the learner's
    selected learning path.
    """

    # Get the authenticated learner ID from the access token.
    user_id = int(current_user["sub"])

    # Build the Tutor Agent workflow.
    graph = build_tutor_graph()

    # Provide only the information needed to start
    # the initial learning path setup.
    initial_state = {
        "user_id": user_id,
        "selected_path": data.selected_path,
    }

    try:
        # Run the Tutor Agent until the initial
        # diagnostic generation flow finishes.
        result = graph.invoke(initial_state)

    except ValueError as exc:
        # Convert validation errors from the agent layer
        # into a client-friendly API response.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "selected_path": result.get("selected_path"),
        "assessment_type": result.get("assessment_type"),
        "diagnostic_topics": result.get(
            "diagnostic_topics",
            [],
        ),
        "assessment_questions": result.get(
            "assessment_questions",
            [],
        ),
        "message": result.get("response"),
    }


@router.post("/diagnostic/submit")
def submit_initial_diagnostic(
    data: DiagnosticSubmitRequest,
    current_user=Depends(get_current_user),
):
    """
    Submit the learner's initial diagnostic answers
    and create the personalized learning path.
    """

    # Get the authenticated learner ID from the access token.
    user_id = int(current_user["sub"])

    # Build the Tutor Agent workflow.
    graph = build_tutor_graph()

    # Build the state required for the diagnostic
    # submission branch of the Tutor Agent.
    initial_state = {
        "user_id": user_id,
        "selected_path": data.selected_path,
        "assessment_questions": data.assessment_questions,
        "assessment_answers": [
            answer.model_dump()
            for answer in data.assessment_answers
        ],
    }

    try:
        # Run the Tutor Agent to evaluate the diagnostic,
        # save mastery, and create the learning path.
        result = graph.invoke(initial_state)

    except ValueError as exc:
        # Convert agent validation errors into
        # a client-friendly API response.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "selected_path": result.get("selected_path"),
        "assessment_result": result.get(
            "assessment_result",
            {},
        ),
        "learning_path": result.get(
            "learning_path",
            {},
        ),
        "current_topic": result.get(
            "current_topic",
            {},
        ),
        "next_topic_id": result.get("next_topic_id"),
        "message": result.get("response"),
    }