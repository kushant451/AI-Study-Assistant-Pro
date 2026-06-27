import json
import re
import random
import time
from rag.citation_engine import chunks_to_plain_text

MODEL = "gemini-2.0-flash"

QUIZ_SYSTEM_PROMPT = """
You are a study assistant that creates quizzes.

Based on the material given, create exactly {num_questions} multiple-choice questions.

Respond with ONLY valid JSON (no markdown, no extra text) in this exact format:

{{
  "questions": [
    {{
      "question": "question text",
      "options": ["option A", "option B", "option C", "option D"],
      "correct_index": 0,
      "explanation": "short explanation of the correct answer"
    }}
  ]
}}
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


def _gemini_call(client, system_prompt, user_prompt):
    """Call Gemini with retry logic using the new google-genai SDK."""
    prompt = f"{system_prompt}\n\n{user_prompt}"
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"[GEMINI ERROR] attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(wait)
    raise Exception("Gemini API failed after all retries.")


def generate_quiz(client, chunks, num_questions=15):
    sample_chunks = random.sample(chunks, min(15, len(chunks)))
    context = chunks_to_plain_text(sample_chunks, limit=15)

    system_prompt = QUIZ_SYSTEM_PROMPT.format(num_questions=num_questions)
    user_prompt = f"Material:\n{context}"

    # ✅ Fixed: was using OpenAI client.chat.completions.create()
    raw = _gemini_call(client, system_prompt, user_prompt)

    print("QUIZ RESPONSE:")
    print(raw)

    questions = parse_quiz_json(raw)
    return questions if questions else []