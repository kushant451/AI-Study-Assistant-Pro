import time

STYLE_PROMPTS = {
    "brief": "Summarize the following study material in 3-4 short bullet points, covering only the most important ideas.",
    "detailed": """
Create complete university exam notes from the document.

Requirements:
- Cover ALL topics from the document.
- Do not skip any chapter.
- Include all major headings.
- Preserve chapter-wise structure.
- Include definitions, features, advantages, disadvantages, applications and examples.
- Do NOT copy text directly.
- Write exam-ready answers (10/15 marks).
"""
}


def detect_style(query):
    query_lower = query.lower()

    if any(word in query_lower for word in [
        "full pdf summary",
        "complete pdf summary",
        "summarize entire pdf"
    ]):
        return "detailed"

    if any(word in query_lower for word in [
        "detail", "detailed", "in depth", "elaborate",
        "full summary", "long summary"
    ]):
        return "detailed"

    if any(word in query_lower for word in [
        "exam", "revision", "important points", "key points"
    ]):
        return "brief"

    return "brief"


def summarize(client, chunks, style="brief", query=""):

    from rag.citation_engine import chunks_to_plain_text

    print("BATCH MODE ACTIVE")

    # auto style detection
    if query:
        style = detect_style(query)

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])

    batch_size = 10
    chunk_batches = [
        chunks[i:i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    batch_summaries = []

    for batch in chunk_batches:

        context = chunks_to_plain_text(batch, limit=len(batch))

        user_prompt = f"""
User Request:
{query}

Material:
{context}
"""

        batch_system_prompt = system_prompt  # IMPORTANT FIX

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": batch_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        batch_summaries.append(response.choices[0].message.content)

        time.sleep(1.2)

    print("STYLE SELECTED:", style)

    combined_summary = "\n\n".join(batch_summaries)

    final_prompt = f"""
Combine all section-wise summaries into one complete, structured, exam-ready document.

Do NOT skip any topic.

CONTENT:
{combined_summary}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.3,
    )

    print("summary_agent loaded successfully")

    return response.choices[0].message.content