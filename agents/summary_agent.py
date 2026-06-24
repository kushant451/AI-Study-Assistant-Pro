STYLE_PROMPTS = {
    "brief": "Summarize the following study material in 3-4 short bullet points, covering only the most important ideas.",
    "detailed": """
Summarize the entire document.

Requirements:
- Cover ALL major topics.
- Use numbered headings and subheadings.
- Preserve chapter-wise structure.
- Include important points under each heading.
- Generate exam-ready notes suitable for 10-mark and 15-mark answers.
- Explain each topic in detail.
- Include definitions, concepts, advantages, disadvantages, applications, and examples where relevant.
- Do not skip any important section.
- Make the summary suitable for university exam preparation.
""",
    "exam": "Summarize the following study material as exam-focused revision notes: key definitions, formulas, and concepts a student is likely to be tested on.",
}

from rag.citation_engine import chunks_to_plain_text


def detect_style(query):
    query_lower = query.lower()

    if any(word in query_lower for word in [
        "detail",
        "detailed",
        "in depth",
        "elaborate",
        "full summary",
        "complete summary",
        "entire pdf",
        "whole pdf"
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


def summarize(client, chunks, style="brief"):

    if style == "detailed":
        context = chunks_to_plain_text(chunks, limit=len(chunks))
    else:
        context = chunks_to_plain_text(chunks)

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
