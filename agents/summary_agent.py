import time

STYLE_PROMPTS = {
    "brief": (
        "You are an expert exam notes writer. "
        "Summarize the given material into clear bullet points. "
        "Cover ALL topics mentioned. Be thorough."
    ),
    "detailed": (
        "You are an expert exam notes writer. "
        "Create detailed exam notes from the given material. "
        "Use numbered headings and sub-headings. "
        "Include definitions, key features, pros/cons, examples. "
        "Cover EVERY concept mentioned. Do not skip anything."
    )
}

MAX_CHUNKS       = 117
CHUNKS_PER_BATCH = 6


def detect_style(query):
    query_lower = query.lower()
    detailed_keywords = [
        "full pdf", "entire pdf", "whole pdf", "complete pdf",
        "full summary", "complete summary", "entire summary",
        "detail", "detailed", "in depth", "elaborate",
        "more", "expand", "long", "exam", "revision",
        "important points", "key points", "notes", "all topics"
    ]
    if any(word in query_lower for word in detailed_keywords):
        return "detailed"
    return "brief"


def gemini_call(client, system_prompt, user_prompt):
    for attempt in range(5):
        try:
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = client.generate_content(prompt)
            return response.text
        except Exception as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"[GEMINI] Error (attempt {attempt+1}/5). Waiting {wait}s... {e}")
            time.sleep(wait)
    raise Exception("Gemini API failed after all retries.")


def safe_trim(text, max_chars):
    return text[:max_chars]


def summarize(client, chunks, style="brief", query=""):
    from rag.citation_engine import chunks_to_plain_text

    print("=" * 60)
    print("SUMMARY AGENT  —  full PDF mode")
    print(f"  Style   : {style}")
    print(f"  Chunks  : {len(chunks)}  |  cap: {MAX_CHUNKS}")
    print(f"  Batch   : {CHUNKS_PER_BATCH} chunks/call")
    print("=" * 60)

    chunks = chunks[:MAX_CHUNKS]
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["detailed"])
    query_trimmed = safe_trim(query, 300)

    batches = [
        chunks[i: i + CHUNKS_PER_BATCH]
        for i in range(0, len(chunks), CHUNKS_PER_BATCH)
    ]
    print(f"  Total batches: {len(batches)}\n")

    all_summaries = []

    for idx, batch in enumerate(batches):
        print(f"[Batch {idx+1}/{len(batches)}]  {len(batch)} chunks")

        chars_per_chunk = 3000 // max(len(batch), 1)
        context_parts = []
        for chunk in batch:
            raw = chunks_to_plain_text([chunk], limit=1)
            context_parts.append(safe_trim(raw, chars_per_chunk))
        context = "\n\n".join(context_parts)

        user_prompt = (
            f"User request: {query_trimmed}\n\n"
            f"--- PDF CONTENT ---\n{context}\n---\n\n"
            "Write complete exam notes for ALL topics in the content above."
        )

        text = gemini_call(client, system_prompt, user_prompt)
        all_summaries.append(text)
        print(f"  [OK] {len(text)} chars returned")

        time.sleep(1)  # small delay

    if not all_summaries:
        return "No content could be summarised."

    print(f"\n✅ Done. {len(all_summaries)} sections collected.")

    header = f"# Exam Notes — Full PDF Summary\n{'='*50}\n\n"
    divider = "\n\n" + "─" * 40 + "\n\n"
    return header + divider.join(all_summaries)