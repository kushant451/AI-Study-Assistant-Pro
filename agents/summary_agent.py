import time
from groq import RateLimitError

STYLE_PROMPTS = {
    "brief": (
        "Summarize the following study material in 3-4 short bullet points, "
        "covering only the most important ideas."
    ),
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
        return "detailed"

    return "brief"


def groq_call(client, **kwargs):
    """
    Retry automatically when Groq returns a rate-limit error.
    """

    retries = 5

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError:
            wait_time = min(2 * (attempt + 1), 10)

            print(
                f"Rate limit hit. "
                f"Retry {attempt + 1}/{retries}. "
                f"Waiting {wait_time}s..."
            )

            time.sleep(wait_time)

    raise Exception("Groq API rate limit exceeded after retries")


def summarize(client, chunks, style="brief", query=""):

    from rag.citation_engine import chunks_to_plain_text

    print("BATCH MODE ACTIVE")

    # Smaller batches reduce TPM spikes
    batch_size = 2

    chunk_batches = [
        chunks[i:i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    batch_summaries = []

    system_prompt = STYLE_PROMPTS.get(
        style,
        STYLE_PROMPTS["brief"]
    )

    for idx, batch in enumerate(chunk_batches):

        print(
            f"Processing batch "
            f"{idx + 1}/{len(chunk_batches)}"
        )

        time.sleep(1)

        context = chunks_to_plain_text(
            batch,
            limit=len(batch)
        )

        # Prevent giant prompts
        context = context[:6000]

        user_prompt = f"""
User Request:
{query}

Material:
{context}
"""

        response = groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
            temperature=0.3,
            max_tokens=500
        )

        summary_text = (
            response.choices[0]
            .message.content
        )

        batch_summaries.append(summary_text)

    print("STYLE SELECTED:", style)

    combined_summary = "\n\n".join(batch_summaries)

    # Prevent huge merge prompt
    MAX_MERGE_CHARS = 12000
    combined_summary = combined_summary[:MAX_MERGE_CHARS]

    final_prompt = f"""
You are given section-wise summaries of a full document.

TASK:
- Merge all sections into ONE complete PDF summary
- Maintain chapter-wise structure
- Do NOT skip any topic
- Do NOT repeat content
- Ensure full coverage of entire document

CONTENT:
{combined_summary}
"""

    print("NUMBER OF BATCHES:", len(batch_summaries))
    print("COMBINED SUMMARY LENGTH:", len(combined_summary))
    print("FINAL PROMPT LENGTH:", len(final_prompt))

    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": final_prompt
            },
        ],
        temperature=0.3,
        max_tokens=1200
    )

    print("summary_agent loaded successfully")
    print("detect_style found:", callable(detect_style))

    return response.choices[0].message.content