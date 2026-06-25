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
        "detail", "detailed", "in depth", "elaborate",
        "full summary", "complete summary", "entire pdf",
        "whole pdf", "more summary", "expand summary",
        "detailed summary", "long summary"
    ]):
        return "detailed"

    if any(word in query_lower for word in [
        "exam", "revision", "important points", "key points"
    ]):
        return "detailed"

    return "brief"


def estimate_tokens(text):
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def groq_call(client, **kwargs):
    retries = 5

    for attempt in range(retries):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError:
            wait_time = min(4 * (attempt + 1), 30)  # longer waits
            print(f"Rate limit hit. Retry {attempt + 1}/{retries}. Waiting {wait_time}s...")
            time.sleep(wait_time)

    raise Exception("Groq API rate limit exceeded after retries")


def summarize(client, chunks, style="brief", query=""):

    from rag.citation_engine import chunks_to_plain_text

    print("=" * 60)
    print("SUMMARY AGENT STARTED")
    print("=" * 60)

    batch_size = 1

    chunk_batches = [
        chunks[i:i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    batch_summaries = []

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])

    # ── Token budget ──────────────────────────────────────────
    TPM_LIMIT        = 6000
    SAFETY_MARGIN    = 1000          # headroom for prompt scaffolding
    MAX_TOKENS_OUT   = 200           # output tokens per batch call
    CONTEXT_TOKEN_BUDGET = TPM_LIMIT - SAFETY_MARGIN - MAX_TOKENS_OUT
    MAX_CONTEXT_CHARS    = CONTEXT_TOKEN_BUDGET * 4  # ≈ 4 chars/token → ~1900
    MAX_CONTEXT_CHARS    = min(MAX_CONTEXT_CHARS, 400)  # hard cap at 400 chars
    # ─────────────────────────────────────────────────────────

    print("STYLE SELECTED:", style)
    print("TOTAL CHUNKS:", len(chunks))
    print("TOTAL BATCHES:", len(chunk_batches))
    print("MAX CONTEXT CHARS PER BATCH:", MAX_CONTEXT_CHARS)

    # =====================================================
    # STEP 1: SUMMARIZE EACH BATCH
    # =====================================================
    for idx, batch in enumerate(chunk_batches):

        print(f"Processing batch {idx + 1}/{len(chunk_batches)}")

        time.sleep(3)  # increased sleep to stay under TPM

        context = chunks_to_plain_text(batch, limit=len(batch))
        context = context[:MAX_CONTEXT_CHARS]  # safe trim

        user_prompt = f"User Request:\n{query}\n\nMaterial:\n{context}"

        # Pre-flight token check
        estimated = estimate_tokens(system_prompt + user_prompt)
        print(f"Estimated input tokens: {estimated}")

        if estimated + MAX_TOKENS_OUT > TPM_LIMIT:
            # Trim context further if still too large
            overage_chars = (estimated + MAX_TOKENS_OUT - TPM_LIMIT) * 4
            context = context[:max(100, len(context) - overage_chars)]
            user_prompt = f"User Request:\n{query}\n\nMaterial:\n{context}"
            print(f"Trimmed context to {len(context)} chars after pre-flight check")

        response = groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS_OUT
        )

        try:
            print("USAGE:", response.usage)
        except Exception:
            pass

        summary_text = response.choices[0].message.content[:800]
        batch_summaries.append(summary_text)

    # =====================================================
    # STEP 2: MERGE BATCH SUMMARIES
    # =====================================================

    combined_summary = "\n\n".join(batch_summaries)

    # Keep merge input well under TPM
    MAX_MERGE_CHARS = 1600  # reduced from 2500
    combined_summary = combined_summary[:MAX_MERGE_CHARS]

    print("AFTER MERGE LENGTH:", len(combined_summary))

    final_prompt = (
        "You are given section-wise summaries of a document.\n\n"
        "TASK:\n"
        "- Merge into clean study notes\n"
        "- Remove duplicates\n"
        "- Keep structure\n"
        "- Be concise and exam-friendly\n\n"
        f"CONTENT:\n{combined_summary}"
    )

    # Pre-flight for merge call
    merge_estimated = estimate_tokens(system_prompt + final_prompt)
    print(f"Merge estimated input tokens: {merge_estimated}")

    time.sleep(3)  # wait before merge call too

    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.3,
        max_tokens=500  # reduced from 600
    )

    try:
        print("FINAL USAGE:", response.usage)
    except Exception:
        pass

    print("SUMMARY COMPLETE")

    return response.choices[0].message.content