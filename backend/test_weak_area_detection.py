from app.tools.weak_area_detection import detect_weak_areas


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
Result: Correct

Question 3:
Concept: Parallel Processing
Learner Answer: Transformers can process tokens in parallel
Correct Answer: Transformers can process tokens in parallel
Result: Correct

Question 4:
Concept: Sequence Relationships
Learner Answer: A storage mechanism
Correct Answer: Relationships between different tokens in a sequence
Result: Correct"""


result = detect_weak_areas.invoke(
    {
        "assessment_results": assessment_results,
    }
)

print(result)
