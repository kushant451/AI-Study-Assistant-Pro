STYLE_PROMPTS = {
    "brief": "Summarize the following study material in 3-4 short bullet points, covering only the most important ideas.",
    "detailed": """
Create complete university exam notes from the document.

Requirements:

- Cover ALL topics from the document.
- Do not skip any chapter.
- Include all major headings found in the document.
- If the document has multiple units, summarize every unit.
- Use numbered main headings.
- Under every heading provide 3-5 subpoints.
- Expand every subpoint in 2-4 lines.
- Preserve chapter-wise flow.
- Include definitions, features, advantages, disadvantages, applications and examples where relevant.
- Do not simply copy text from the document.
- Rewrite content as study notes.
- Make every topic suitable for 10-mark and 15-mark university answers.

Format:

1. Topic Name
   - Point 1
   - Point 2
   - Point 3
   - Point 4

2. Topic Name
   - Point 1
   - Point 2
   - Point 3
   - Point 4

Continue until all topics are covered.
""",
    "exam": "Summarize the following study material as exam-focused revision notes: key definitions, formulas, and concepts a student is likely to be tested on.",
}

from rag.citation_engine import chunks_to_plain_text


def detect_style(query):
    query_lower = query.lower()
    if any(word in query_lower for word in [
        "full pdf summary",
        "summarize entire pdf",
        "summarize complete pdf",
        "complete pdf summary"
    ]):
        return "detailed"
    if any(word in query_lower for word in [
        "detail",
        "detailed",
        "in depth",
        "elaborate",
        "full summary",
        "complete summary",
        "entire pdf",
        "whole pdf",
        "more summary",
        "expand summary",
        "detailed summary",
        "long summary"
    ]):
        return "detailed"

    if any(word in query_lower for word in [
        "exam",
        "revision",
        "important points",
        "key points"
    ]):
        return "exam"

    return "brief"


def summarize(client, chunks, style="brief", query=""):

    context = chunks_to_plain_text(
        chunks,
        limit=len(chunks)
    )

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])

    user_prompt = f"""
    User Request:
    {query}

    Material:
    {context}
    """

    print("STYLE:", style)
    print("CHUNKS:", len(chunks))
    print("CONTEXT LENGTH:", len(context))

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as e:
        print("GROQ ERROR:", e)
        raise

    return response.choices[0].message.content