def evaluate_quiz(questions, user_answers):
    score = 0
    details = []

    for i, q in enumerate(questions):
        selected = user_answers.get(i)
        correct = q["correct_index"]
        is_correct = selected == correct

        if is_correct:
            score += 1

        details.append({
            "question": q["question"],
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "options": q["options"],
        })

    return {"score": score, "total": len(questions), "details": details}


def quiz_to_text(questions):
    lines = []
    for i, q in enumerate(questions):
        lines.append(f"Q{i+1}. {q['question']}")
        for j, opt in enumerate(q["options"]):
            marker = "*" if j == q["correct_index"] else " "
            lines.append(f"   {marker} {opt}")
        lines.append(f"   Explanation: {q.get('explanation','')}")
        lines.append("")
    return "\n".join(lines)