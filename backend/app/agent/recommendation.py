def determine_recommended_action(
    mastery_score: float | None
) -> str:
    """
    Determine the recommended learning action
    from the learner's latest mastery score.

    This function provides a single source of truth
    for recommendation rules across the system.
    """

    # If the learner has not completed an assessment yet,
    # start by explaining the topic.
    if mastery_score is None:
        return "explain"

    # Very low mastery requires explanation.
    if mastery_score < 40:
        return "explain"

    # Moderate mastery requires additional practice.
    if mastery_score < 70:
        return "practice"

    # Good mastery requires review before progressing.
    if mastery_score < 85:
        return "review"

    # High mastery means the learner is ready
    # for the next learning step.
    return "recommend"



def build_recommendation(
    mastery_score: float | None,
    weak_areas: list | None = None
) -> dict:
    """
    Build a detailed recommendation using the learner's
    mastery score and identified weak areas.

    The recommendation includes both the next learning
    action and the reason behind that decision.
    """

    # Reuse the shared recommendation rules
    # to determine the learner's next action.
    recommended_action = determine_recommended_action(
        mastery_score=mastery_score
    )

    # Normalize weak areas so the recommendation logic
    # can safely work when no weak areas are available.
    weak_areas = weak_areas or []

    # The learner has not completed an assessment yet.
    if mastery_score is None:
        reason = (
            "No assessment result is available for this topic. "
            "The learner should start with an explanation."
        )

    # Very low mastery indicates that the learner
    # needs stronger conceptual understanding.
    elif mastery_score < 40:
        reason = (
            f"Mastery is {mastery_score}%. "
            "The learner needs additional explanation "
            "before progressing."
        )

    # Moderate mastery indicates that the learner
    # should strengthen understanding through practice.
    elif mastery_score < 70:
        reason = (
            f"Mastery is {mastery_score}%. "
            "The learner understands some of the topic "
            "but needs additional practice."
        )

    # Good mastery indicates that the learner
    # should review remaining gaps before progressing.
    elif mastery_score < 85:
        if weak_areas:
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner should review the identified "
                "weak areas before progressing."
            )
        else:
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner should review the topic "
                "before progressing."
            )

    # High mastery means the learner can progress.
    # Remaining weak areas are kept as supplemental
    # information rather than blocking progression.
    elif weak_areas:
        reason = (
            f"Mastery is {mastery_score}%. "
            "The learner is ready to progress, but a short "
            "review of the remaining weak areas is recommended."
        )

    else:
        reason = (
            f"Mastery is {mastery_score}%. "
            "The learner has demonstrated strong mastery "
            "and is ready for the next learning step."
        )

    return {
        "recommended_action": recommended_action,
        "recommendation_reason": reason,
    }