STYLE_PROMPTS = {
    "brief": "Summarize the following study material in 3-4 short bullet points, covering only the most important ideas.",
    "detailed": "Summarize the following study material in detail, organized under clear topic headings, covering all major concepts.",
    "exam": "Summarize the following study material as exam-focused revision notes: key definitions, formulas, and concepts a student is likely to be tested on.",
}

from rag.citation_engine import chunks_to_plain_text


def detect_style(query):
    query_lower = query.lower()
    if any(word in query_lower for word in ["detail", "detailed", "in depth", "elaborate"]):
        return "detailed"
    if any(word in query_lower for word in ["exam", "revision", "important points", "key points"]):
        return "exam"
    return "brief"


def summarize(client, chunks, style="brief"):
    context = chunks_to_plain_text(chunks, limit=8)

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])
    user_prompt = f"Material:\n{context}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content