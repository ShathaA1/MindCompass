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