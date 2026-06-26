import time
import re
from groq import RateLimitError, APIStatusError
 
STYLE_PROMPTS = {
    "brief": "Summarize in 5-6 bullet points covering only key ideas. Be concise.",
    "detailed": (
        "Create concise exam notes. Use numbered headings. "
        "3-5 bullet points each. Include definitions, pros/cons, uses. "
        "No repetition. Be brief but complete."
    )
}
 
TPM_HARD_LIMIT  = 6000
OUTPUT_TOKENS   = 400   # was 150 — too small for useful summaries
PROMPT_OVERHEAD = 100
MAX_CHUNKS      = 30    # was 5 — now covers much more of the PDF
 
# How many chunks to combine into one batch before calling the API.
# Batching = fewer API calls = faster + less rate limiting.
CHUNKS_PER_BATCH = 5
 
 
def detect_style(query):
    query_lower = query.lower()
    detailed_keywords = [
        "full pdf summary", "summarize entire pdf", "summarize complete pdf",
        "complete pdf summary", "detail", "detailed", "in depth", "elaborate",
        "full summary", "complete summary", "entire pdf", "whole pdf",
        "more summary", "expand summary", "detailed summary", "long summary",
        "exam", "revision", "important points", "key points"
    ]
    if any(word in query_lower for word in detailed_keywords):
        return "detailed"
    return "brief"
 
 
def count_tokens_approx(text):
    return len(text) // 3
 
 
def safe_trim(text, max_chars):
    return text[:max_chars]
 
 
def extract_wait_time(error_message):
    match = re.search(r'try again in ([0-9.]+)s', str(error_message))
    return float(match.group(1)) + 2.0 if match else 15.0
 
 
def groq_call(client, **kwargs):
    for attempt in range(8):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIStatusError) as e:
            wait = max(extract_wait_time(e), 8.0)
            print(f"[GROQ] Error (attempt {attempt+1}/8). Waiting {wait:.1f}s... {e}")
            time.sleep(wait)
    raise Exception("Groq API failed after all retries.")
 
 
def summarize(client, chunks, style="brief", query=""):
    from rag.citation_engine import chunks_to_plain_text
 
    print("=" * 60)
    print("SUMMARY AGENT STARTED")
    print(f"  Style: {style} | Total chunks: {len(chunks)}")
 
    chunks = chunks[:MAX_CHUNKS]
    print(f"  Processing: {len(chunks)} chunks (capped at {MAX_CHUNKS})")
    print("=" * 60)
 
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])
    query_trimmed = query[:150]
    batch_summaries = []
 
    # ── BATCHING: group CHUNKS_PER_BATCH chunks together ──────────────────
    # Instead of 1 API call per chunk (slow + hits rate limits fast),
    # we combine multiple chunks into a single prompt.
    batches = [
        chunks[i : i + CHUNKS_PER_BATCH]
        for i in range(0, len(chunks), CHUNKS_PER_BATCH)
    ]
 
    for batch_idx, batch in enumerate(batches):
        print(f"\n[BATCH {batch_idx+1}/{len(batches)}]  ({len(batch)} chunks)")
 
        # Combine all chunks in this batch into one context block
        combined_context = ""
        for chunk in batch:
            piece = chunks_to_plain_text([chunk], limit=1)
            combined_context += safe_trim(piece, 600) + "\n\n"  # was 200 per chunk
 
        combined_context = safe_trim(combined_context, 2500)
 
        user_prompt = f"Request: {query_trimmed}\n\nMaterial:\n{combined_context}"
 
        total_est = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS
        print(f"  [TOKEN CHECK] ~{total_est} / {TPM_HARD_LIMIT}")
 
        if total_est > TPM_HARD_LIMIT:
            # Context too large — trim and retry
            combined_context = safe_trim(combined_context, 1200)
            user_prompt = f"Request: {query_trimmed}\n\nMaterial:\n{combined_context}"
            total_est = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS
            print(f"  [TRIMMED TOKEN CHECK] ~{total_est} / {TPM_HARD_LIMIT}")
 
        if total_est > TPM_HARD_LIMIT:
            print(f"  [SKIP] Still too large after trim: {total_est} tokens")
            continue
 
        response = groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=OUTPUT_TOKENS,
        )
 
        try:
            print(f"  [USAGE] {response.usage}")
        except Exception:
            pass
 
        batch_summaries.append(response.choices[0].message.content[:800])
 
        # Adaptive sleep: only sleep between batches, not between every chunk.
        # 6s is usually enough to stay under TPM limits.
        if batch_idx < len(batches) - 1:
            time.sleep(6)
 
    if not batch_summaries:
        return "No content could be summarised."
 
    # ── MERGE STEP ─────────────────────────────────────────────────────────
    print("\n[MERGE STEP]")
    time.sleep(6)
 
    combined = "\n\n".join(batch_summaries)
    combined = safe_trim(combined, 3000)
 
    merge_prompt = (
        "Merge the notes below into clean, structured exam-friendly study notes. "
        "Use numbered headings. Remove duplicates. Keep all unique key points. "
        "Be concise but thorough.\n\n"
        f"CONTENT:\n{combined}"
    )
 
    total_est = count_tokens_approx(system_prompt + merge_prompt) + OUTPUT_TOKENS
    print(f"  [MERGE TOKEN CHECK] ~{total_est} / {TPM_HARD_LIMIT}")
 
    if total_est > TPM_HARD_LIMIT:
        print("  [MERGE] Too large, returning combined batch summaries directly.")
        return combined
 
    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": merge_prompt},
        ],
        temperature=0.3,
        max_tokens=OUTPUT_TOKENS,
    )
 
    print("\nSUMMARY COMPLETE")
    return response.choices[0].message.content