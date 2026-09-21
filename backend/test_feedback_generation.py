from app.tools.feedback_generation import generate_feedback


assessment_results = """
Question 1:
Concept: Transformers
Learner Answer: A type of neural network architecture
Correct Answer: A type of neural network architecture
Result: Correct

Question 2:
Concept: Attention Mechanism
Learner Answer: A sorting mechanism
Correct Answer: An attention mechanism
Result: Incorrect

Question 3:
Concept: Parallel Processing
Learner Answer: Transformers can process tokens in parallel
Correct Answer: Transformers can process tokens in parallel
Result: Correct

Question 4:
Concept: Sequence Relationships
Learner Answer: A storage mechanism
Correct Answer: Relationships between different tokens in a sequence
Result: Incorrect
"""

weak_areas = """
[
    {
        "area": "Attention Mechanism",
        "reason": "The learner incorrectly identified an attention mechanism as a sorting mechanism."
    },
    {
        "area": "Sequence Relationships",
        "reason": "The learner incorrectly identified sequence relationships as a storage mechanism."
    }
]
"""

student_level = "beginner"


result = generate_feedback.invoke(
    {
        "assessment_results": assessment_results,
        "weak_areas": weak_areas,
        "student_level": student_level,
    }
)

print(result)