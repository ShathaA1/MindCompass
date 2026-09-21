from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import get_db
from app.database.models import LearnerProfile
from app.schemas.learner import OnboardingRequest


router = APIRouter(prefix="/learners", tags=["Learners"])


@router.post("/onboarding")
def onboarding(
    data: OnboardingRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = int(current_user["sub"])

    existing_profile = (
        db.query(LearnerProfile)
        .filter(LearnerProfile.user_id == user_id)
        .first()
    )

    if existing_profile:
        raise HTTPException(
            status_code=400,
            detail="Learner profile already exists",
        )

    profile = LearnerProfile(
        user_id=user_id,
        goal=data.goal,
        initial_level=data.initial_level,
        weekly_hours=data.weekly_hours,
        current_level=data.initial_level,
        preferred_format=data.preferred_format,
        preferred_pace=data.preferred_pace,
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)

    return {
        "message": "Onboarding completed successfully",
        "profile_id": profile.profile_id,
    }


@router.get("/profile")
def get_profile(
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

    return {
        "profile_id": profile.profile_id,
        "user_id": profile.user_id,
        "goal": profile.goal,
        "initial_level": profile.initial_level,
        "current_level": profile.current_level,
        "weekly_hours": profile.weekly_hours,
        "preferred_format": profile.preferred_format,
        "preferred_pace": profile.preferred_pace,
        "updated_at": profile.updated_at,
    }


@router.put("/profile")
def update_profile(
    data: OnboardingRequest,
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

    profile.goal = data.goal
    profile.initial_level = data.initial_level
    profile.weekly_hours = data.weekly_hours
    profile.current_level = data.initial_level
    profile.preferred_format = data.preferred_format
    profile.preferred_pace = data.preferred_pace

    db.commit()
    db.refresh(profile)

    return {
        "message": "Profile updated successfully",
        "profile_id": profile.profile_id,
    }