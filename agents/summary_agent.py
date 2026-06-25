import time
from groq import RateLimitError

STYLE_PROMPTS = {
    "brief": (
        "Summarize the following study material in 3-4 short bullet points, "
        "covering only the most important ideas."
    ),

    "detailed": """
Create university exam notes.

Requirements:
- Cover all important topics.
- Use numbered headings.
- Provide 3-5 key points per heading.
- Include definitions, features, advantages,
  disadvantages and applications where relevant.
- Preserve topic-wise structure.
- Rewrite as study notes.
- Avoid repetition.
- Keep concise but complete.

Format:

1. Topic Name
   - Point 1
   - Point 2
   - Point 3

2. Topic Name
   - Point 1
   - Point 2
   - Point 3
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

    print("=" * 60)
    print("SUMMARY AGENT STARTED")
    print("=" * 60)

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

    print("STYLE SELECTED:", style)
    print("TOTAL CHUNKS:", len(chunks))
    print("TOTAL BATCHES:", len(chunk_batches))

    # =====================================================
    # STEP 1: SUMMARIZE EACH BATCH
    # =====================================================

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

        # Reduced from 6000 -> 2000
        context = context[:2000]

        user_prompt = f"""
User Request:
{query}

Material:
{context}
"""

        print("CONTEXT LENGTH:", len(context))
        print("USER PROMPT LENGTH:", len(user_prompt))

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
            max_tokens=250
        )

        try:
            print("USAGE:", response.usage)
        except Exception:
            pass

        summary_text = (
            response.choices[0]
            .message.content
        )

        # Limit size of each batch summary
        summary_text = summary_text[:1200]

        batch_summaries.append(summary_text)

    # =====================================================
    # STEP 2: MERGE BATCH SUMMARIES
    # =====================================================

    combined_summary = "\n\n".join(batch_summaries)
    print("BEFORE TRUNCATION:", len(combined_summary))

    MAX_MERGE_CHARS = 5000

    combined_summary = combined_summary[:MAX_MERGE_CHARS]

    print("AFTER TRUNCATION:", len(combined_summary))

    # Reduced from 12000 -> 5000
    MAX_MERGE_CHARS = 5000
    combined_summary = combined_summary[:MAX_MERGE_CHARS]

    final_prompt = f"""
You are given section-wise summaries of a full document.

TASK:
- Merge all sections into one structured study note.
- Maintain chapter-wise flow.
- Remove duplicate content.
- Keep important headings.
- Do not invent information.
- Keep it concise and exam-friendly.

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
        max_tokens=600
    )

    try:
        print("FINAL USAGE:", response.usage)
    except Exception:
        pass

    print("SUMMARY COMPLETE")

    return response.choices[0].message.content