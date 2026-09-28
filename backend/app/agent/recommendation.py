def determine_recommended_action(
    mastery_score: float | None,
    last_action: str | None = None,
) -> str:
    """
    Determine the learner's next recommended action
    using mastery and the previously completed action.
    """

    # Before the first assessment, guide the learner
    # through explain -> practice -> assess.
    if mastery_score is None:
        if last_action == "explain":
            return "practice"

        if last_action == "practice":
            return "assess"

        return "explain"

    # Very low mastery requires stronger support.
    if mastery_score < 40:
        if last_action == "explain":
            return "practice"

        if last_action == "practice":
            return "assess"

        return "explain"

    # Moderate mastery should alternate between
    # practice and reassessment.
    if mastery_score < 70:
        if last_action == "practice":
            return "assess"

        return "practice"

    # Good mastery should alternate between
    # focused review and reassessment.
    if mastery_score < 80:
        if last_action == "review":
            return "assess"

        return "review"

    # Mastery of 80% or higher means the learner
    # is ready to continue to the next topic.
    return "recommend"



def build_recommendation(
    mastery_score: float | None,
    weak_areas: list | None = None,
    last_action: str | None = None,
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
        mastery_score=mastery_score,
        last_action=last_action,
    )

    # Normalize weak areas so the recommendation logic
    # can safely work when no weak areas are available.
    weak_areas = weak_areas or []

    # Explain why the selected next action is appropriate.
    if mastery_score is None:
        if recommended_action == "practice":
            reason = (
                "The learner has completed the explanation "
                "and should now reinforce understanding through practice."
            )
        elif recommended_action == "assess":
            reason = (
                "The learner has completed the explanation and practice "
                "and is ready for an assessment."
            )
        else:
            reason = (
                "No assessment result is available for this topic. "
                "The learner should start with an explanation."
            )

    elif mastery_score < 40:
        if recommended_action == "practice":
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner has reviewed the explanation and should "
                "now reinforce understanding through practice."
            )
        elif recommended_action == "assess":
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner has completed additional practice "
                "and should now be reassessed."
            )
        else:
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner needs additional explanation "
                "before progressing."
            )

    elif mastery_score < 70:
        if recommended_action == "assess":
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner has completed additional practice "
                "and should now be reassessed."
            )
        else:
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner understands some of the topic "
                "but needs additional practice."
            )

    elif mastery_score < 80:
        if recommended_action == "assess":
            reason = (
                f"Mastery is {mastery_score}%. "
                "The learner has completed the review "
                "and should now be reassessed."
            )
        elif weak_areas:
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
