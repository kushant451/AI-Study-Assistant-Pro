import json
import re
import random
from rag.citation_engine import chunks_to_plain_text

QUIZ_SYSTEM_PROMPT = """
You are a study assistant that creates quizzes.

Based on the material given, create exactly {num_questions} multiple-choice questions.

Respond with ONLY valid JSON (no markdown, no extra text) in this exact format:

{
  "questions": [
    {
      "question": "question text",
      "options": ["option A", "option B", "option C", "option D"],
      "correct_index": 0,
      "explanation": "short explanation of the correct answer"
    }
  ]
}
"""


def parse_quiz_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    json_text = match.group(0) if match else raw_text

    try:
        data = json.loads(json_text)
        questions = data.get("questions", [])
        if questions:
            return questions
    except (json.JSONDecodeError, AttributeError):
        pass

    return None


def generate_quiz(client, chunks, num_questions=15):
    sample_chunks = random.sample(
        chunks,
        min(15, len(chunks))
    )

    context = chunks_to_plain_text(
        sample_chunks,
        limit=15
    )

    prompt = QUIZ_SYSTEM_PROMPT.format(
        num_questions=num_questions
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Material:\n{context}"},
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content

    print("QUIZ RESPONSE:")
    print(raw)

    questions = parse_quiz_json(raw)

    return questions if questions else []