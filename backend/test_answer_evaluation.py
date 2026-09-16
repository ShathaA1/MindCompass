from app.tools.answer_evaluation import evaluate_answer


question = "What mechanism do Transformers use to identify relationships between tokens?"

correct_answer = "An attention mechanism"

learner_answer = "  an attention mechanism  "
result = evaluate_answer.invoke(
    {
        "question": question,
        "correct_answer": correct_answer,
        "learner_answer": learner_answer,
    }
)

print(result)