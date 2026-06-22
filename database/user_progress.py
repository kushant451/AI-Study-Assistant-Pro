from datetime import datetime
from database.db import get_connection


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_quiz_attempt(score, total, topic=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO quiz_attempts (timestamp, score, total, topic) VALUES (?, ?, ?, ?)",
        (_now(), score, total, topic),
    )
    conn.commit()
    conn.close()


def log_interview_attempt(question, answer, rating=None, feedback=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO interview_attempts (timestamp, question, answer, rating, feedback) VALUES (?, ?, ?, ?, ?)",
        (_now(), question, answer, rating, feedback),
    )
    conn.commit()
    conn.close()


def log_document_processed(filename, word_count, chunk_count):
    conn = get_connection()
    conn.execute(
        "INSERT INTO documents_processed (timestamp, filename, word_count, chunk_count) VALUES (?, ?, ?, ?)",
        (_now(), filename, word_count, chunk_count),
    )
    conn.commit()
    conn.close()


def get_quiz_history(limit=20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM quiz_attempts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_interview_history(limit=20):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM interview_attempts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_progress_summary():
    conn = get_connection()

    quiz_row = conn.execute(
        "SELECT COUNT(*) as attempts, AVG(CAST(score AS FLOAT) / total) as avg_ratio, "
        "SUM(score) as total_correct, SUM(total) as total_questions "
        "FROM quiz_attempts"
    ).fetchone()

    interview_row = conn.execute(
        "SELECT COUNT(*) as attempts, AVG(rating) as avg_rating FROM interview_attempts"
    ).fetchone()

    docs_row = conn.execute(
        "SELECT COUNT(*) as docs_processed FROM documents_processed"
    ).fetchone()

    conn.close()

    avg_quiz_pct = round((quiz_row["avg_ratio"] or 0) * 100, 1)
    avg_interview_rating = round(interview_row["avg_rating"] or 0, 1)

    return {
        "quiz_attempts": quiz_row["attempts"] or 0,
        "avg_quiz_percentage": avg_quiz_pct,
        "total_questions_answered": quiz_row["total_questions"] or 0,
        "interview_attempts": interview_row["attempts"] or 0,
        "avg_interview_rating": avg_interview_rating,
        "documents_processed": docs_row["docs_processed"] or 0,
    }