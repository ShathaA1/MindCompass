from app.tools.practice_generation import generate_practice


context = """
Transformers are a type of neural network architecture commonly used
in natural language processing. They use an attention mechanism to
identify relationships between different tokens in a sequence.
Unlike traditional recurrent neural networks, Transformers can process
tokens in parallel.
"""

weak_areas = """
[
    {
        "area": "Attention Mechanism",
        "reason": "The learner had difficulty identifying the purpose of the attention mechanism."
    }
]
"""


result = generate_practice.invoke(
    {
        "topic": "Transformers",
        "context": context,
        "student_level": "beginner",
        "practice_type": "true_false",
        "num_items": 5,
        "weak_areas": weak_areas,
    }
)

print(result)