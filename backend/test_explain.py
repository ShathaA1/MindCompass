from app.tools.explain import explain


def main():
    result = explain.invoke(
        {
            "topic": "Transformers",
            "student_level": "beginner",
            "context": """
Transformers are neural network architectures that use
attention mechanisms to model relationships between tokens.
They can process relationships between tokens without
depending on recurrent processing.
""",
        }
    )

    print("\n--- EXPLANATION ---\n")
    print(result)


if __name__ == "__main__":
    main()