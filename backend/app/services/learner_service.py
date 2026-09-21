from sqlalchemy.orm import Session

from app.database.models import User, LearnerProfile


def get_learner_context(
    db: Session,
    user_id: int
) -> dict:
    """
    Get the learner's profile information from the database.
    """

    # Find the learner and join the related learner profile
    learner = (
        db.query(User)
        .join(LearnerProfile)
        .filter(User.user_id == user_id)
        .first()
    )

    # Return an empty dictionary if the learner
    # or learner profile cannot be found
    if not learner or not learner.learner_profile:
        return {}

    # Access the learner profile
    profile = learner.learner_profile

    # Return the learner information in a simple dictionary
    # that can be stored inside TutorState
    return {
        "user_id": learner.user_id,
        "name": learner.name,
        "goal": profile.goal,
        "initial_level": profile.initial_level,
        "current_level": profile.current_level,
        "weekly_hours": profile.weekly_hours,
        "preferred_format": profile.preferred_format,
        "preferred_pace": profile.preferred_pace,
    }