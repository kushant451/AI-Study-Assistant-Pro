import time
from groq import RateLimitError
 
# ── Shorten the detailed prompt significantly ──────────────
STYLE_PROMPTS = {
    "brief": "Summarize in 3-4 bullet points covering only key ideas.",
 
    "detailed": (
        "Create concise exam notes. Use numbered headings. "
        "3-5 bullet points each. Include definitions, pros/cons, uses. "
        "No repetition. Be brief but complete."
    )
}
 
# ── Token constants (llama-3.1-8b-instant on Groq free tier) ──
TPM_HARD_LIMIT   = 6000
OUTPUT_TOKENS    = 180   # tokens reserved for output
PROMPT_OVERHEAD  = 120   # tokens for system prompt + scaffolding
MAX_INPUT_TOKENS = TPM_HARD_LIMIT - OUTPUT_TOKENS - PROMPT_OVERHEAD  # = 5700
MAX_CONTEXT_CHARS = MAX_INPUT_TOKENS * 3  # conservative: 3 chars/token → 17100
MAX_CONTEXT_CHARS = min(MAX_CONTEXT_CHARS, 350)  # absolute hard cap
 
 
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
    """Conservative estimate: 1 token per 3 chars."""
    return len(text) // 3
 
 
def safe_trim(text, max_tokens):
    """Trim text to fit within max_tokens (approx)."""
    max_chars = max_tokens * 3
    return text[:max_chars]
 
 
def groq_call(client, **kwargs):
    """Call Groq with exponential backoff on rate limit."""
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = min(5 * (attempt + 1), 30)
            print(f"[GROQ] Rate limit. Retry {attempt+1}/6 in {wait}s...")
            time.sleep(wait)
    raise Exception("Groq API rate limit exceeded after all retries.")
 
 
def build_and_validate_messages(system_prompt, user_prompt, max_tokens_out):
    """
    Build messages dict and verify total token estimate is under limit.
    Returns messages dict, or raises if still too large.
    """
    total_est = count_tokens_approx(system_prompt + user_prompt) + max_tokens_out
    print(f"  [TOKEN CHECK] Estimated total: {total_est} / {TPM_HARD_LIMIT}")
 
    if total_est > TPM_HARD_LIMIT:
        raise ValueError(
            f"Request still too large: ~{total_est} tokens estimated. "
            f"Reduce context further."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
 
 
def summarize(client, chunks, style="brief", query=""):
 
    from rag.citation_engine import chunks_to_plain_text
 
    print("=" * 60)
    print("SUMMARY AGENT STARTED")
    print(f"  Style: {style} | Chunks: {len(chunks)}")
    print(f"  Max context chars/batch: {MAX_CONTEXT_CHARS}")
    print("=" * 60)
 
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])
 
    # Trim query so it never contributes too many tokens
    query_trimmed = query[:200]
 
    batch_summaries = []
 
    # =====================================================
    # STEP 1: Summarise each chunk individually
    # =====================================================
    for idx, chunk in enumerate(chunks):
        print(f"\n[BATCH {idx+1}/{len(chunks)}]")
        time.sleep(4)  # spread calls across the minute
 
        # Get raw text and hard-trim
        context = chunks_to_plain_text([chunk], limit=1)
        context = safe_trim(context, max_tokens=300)  # 300 tokens max for context
 
        user_prompt = (
            f"Request: {query_trimmed}\n\n"
            f"Material:\n{context}"
        )
 
        print(f"  context chars={len(context)} | user_prompt chars={len(user_prompt)}")
 
        try:
            messages = build_and_validate_messages(system_prompt, user_prompt, OUTPUT_TOKENS)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue
 
        response = groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=OUTPUT_TOKENS,
        )
 
        try:
            print(f"  [USAGE] {response.usage}")
        except Exception:
            pass
 
        text = response.choices[0].message.content
        # Trim output too so merge step stays safe
        batch_summaries.append(text[:600])
 
    if not batch_summaries:
        return "No content could be summarised (all chunks were too large)."
 
    # =====================================================
    # STEP 2: Merge all batch summaries into final notes
    # =====================================================
    print("\n[MERGE STEP]")
    time.sleep(4)
 
    # Each batch summary was capped at 600 chars.
    # Fit as many as possible within merge budget.
    merge_token_budget = TPM_HARD_LIMIT - OUTPUT_TOKENS - PROMPT_OVERHEAD
    combined = ""
    for s in batch_summaries:
        candidate = combined + "\n\n" + s if combined else s
        if count_tokens_approx(candidate) <= merge_token_budget:
            combined = candidate
        else:
            print("  [MERGE] Token budget reached — dropping remaining summaries.")
            break
 
    print(f"  Combined length: {len(combined)} chars")
 
    merge_prompt = (
        "Merge these section summaries into clean, exam-friendly study notes. "
        "Remove duplicates. Keep structure. Be concise.\n\n"
        f"CONTENT:\n{combined}"
    )
 
    try:
        messages = build_and_validate_messages(system_prompt, merge_prompt, OUTPUT_TOKENS)
    except ValueError as e:
        print(f"  [MERGE FALLBACK] {e} — returning combined summaries directly.")
        return combined
 
    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        max_tokens=OUTPUT_TOKENS,
    )
 
    try:
        print(f"  [FINAL USAGE] {response.usage}")
    except Exception:
        pass
 
    print("\nSUMMARY COMPLETE")
    return response.choices[0].message.content