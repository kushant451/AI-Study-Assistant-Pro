from database.user_progress import get_quiz_history


def get_quiz_trend():
    history = get_quiz_history(limit=50)
    history.reverse()

    trend = []
    for i, attempt in enumerate(history, start=1):
        pct = round((attempt["score"] / attempt["total"]) * 100, 1) if attempt["total"] else 0
        trend.append({
            "attempt_number": i,
            "percentage": pct,
            "timestamp": attempt["timestamp"],
        })

    return trend


def get_best_and_worst():
    history = get_quiz_history(limit=50)
    if not history:
        return None, None

    scored = [
        {
            **h,
            "percentage": round((h["score"] / h["total"]) * 100, 1) if h["total"] else 0,
        }
        for h in history
    ]

    best = max(scored, key=lambda x: x["percentage"])
    worst = min(scored, key=lambda x: x["percentage"])

    return best, worst