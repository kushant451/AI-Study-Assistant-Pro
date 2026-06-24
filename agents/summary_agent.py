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

IMPORTANT:

- Create a separate heading for every topic found in the document.
- Do not merge multiple topics into one section.
- Do not generate page-wise summaries.
- Generate topic-wise study notes.
- Cover every unit, chapter and heading present in the document.
- If 20 topics exist, create 20 topic headings.
- Do not create a final conclusion section unless it exists in the document.
- Stop only after all topics are covered.
"""
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

    batch_size = 20

    chunk_batches = [
        chunks[i:i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    batch_summaries = []

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])

    for batch in chunk_batches:
        

        context = chunks_to_plain_text(
            batch,
            limit=len(batch)
        )

        user_prompt = f"""
        User Request:
        {query}

        Material:
        {context}
        """
        batch_system_prompt = """
        Summarize this section briefly.
        Capture all important headings and concepts.
        Maximum 300 words.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": batch_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        batch_summaries.append(
            response.choices[0].message.content
        )

    

    print("STYLE SELECTED:", style)


    combined_summary = "\n\n".join(batch_summaries)

    final_prompt = f"""
    Combine the following partial summaries into one complete,
    well-structured study note.

    {combined_summary}
    """
    print("NUMBER OF BATCHES:", len(batch_summaries))
    print("COMBINED SUMMARY LENGTH:", len(combined_summary))
    print("FINAL PROMPT LENGTH:", len(final_prompt))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content