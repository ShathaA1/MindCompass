from app.tools.quiz_generation import generate_quiz


context = """
Transformers are a type of neural network architecture used in
natural language processing. They use an attention mechanism to
identify relationships between different tokens in a sequence.
Unlike traditional recurrent neural networks, Transformers can
process tokens in parallel.
"""


result = generate_quiz.invoke(
    {
        "topic": "Transformers",
        "context": context,
        "student_level": "beginner",
        "num_questions": 5,
        "assessment_type": "placement",
    }
)

print(result)
