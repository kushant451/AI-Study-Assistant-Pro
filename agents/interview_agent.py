import json
import re
from rag.citation_engine import chunks_to_plain_text

QUESTION_PROMPT = """
You are a senior technical interviewer.

Generate {n} unique interview questions from the study material.

Requirements:
- Do not repeat questions.
- Generate fresh questions every time.
- 30% Easy questions
- 40% Medium questions
- 30% Hard questions
- Include conceptual questions
- Include scenario-based questions
- Include compare-and-contrast questions
- Include problem-solving questions
- Focus on understanding, reasoning, and practical application.

Respond with ONLY valid JSON in this format:

{{
    "questions": [
        "Question 1",
        "Question 2",
        "Question 3"
    ]
}}

Material:
{context}
"""

EVALUATION_PROMPT = """
You are a senior technical interviewer evaluating a candidate.

Question:
{question}

Candidate Answer:
{answer}

Evaluate:
- Technical accuracy
- Completeness
- Clarity
- Practical understanding

Rate the answer from 1 to 5:

5 = Excellent
4 = Good
3 = Average
2 = Weak
1 = Poor

Provide constructive feedback explaining:
- What was correct
- What was missing
- How the answer could be improved

Respond with ONLY valid JSON:

{{
    "rating": 4,
    "feedback": "Detailed feedback here"
}}
"""

def _extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    json_text = match.group(0) if match else raw_text

    try:
        return json.loads(json_text)
    except Exception:
        return None

def generate_interview_questions(client, chunks, n=30):
    context = chunks_to_plain_text(chunks, limit=15)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": QUESTION_PROMPT.format(
                    n=n,
                    context=context
                )
            }
        ],
        temperature=0.9,
    )

    data = _extract_json(response.choices[0].message.content)

    if data and "questions" in data:
        questions = data["questions"]

        unique_questions = []
        seen = set()

        for q in questions:
            q_clean = q.strip().lower()

            if q_clean not in seen:
                seen.add(q_clean)
                unique_questions.append(q)

        return unique_questions[:n]

    return []

def evaluate_answer(client, question, answer):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": EVALUATION_PROMPT.format(
                    question=question,
                    answer=answer
                )
            }
        ],
        temperature=0.3,
    )

    data = _extract_json(response.choices[0].message.content)

    if data and "rating" in data:
        return {
            "rating": int(data["rating"]),
            "feedback": data.get("feedback", "")
        }

    return {
        "rating": None,
        "feedback": "Could not evaluate this answer automatically."
    }