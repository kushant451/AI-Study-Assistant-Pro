import time
import re
from groq import RateLimitError, APIStatusError
 
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
 
TPM_HARD_LIMIT   = 5500
PROMPT_OVERHEAD  = 200
MAX_CHUNKS       = 117       # process ALL chunks
CHUNKS_PER_BATCH = 6         # chunks combined per API call
OUTPUT_TOKENS    = 1024      # max per batch summary
SLEEP_BETWEEN    = 13        # seconds between calls (Groq free tier safe)
 
 
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
 
 
def count_tokens_approx(text):
    return len(text) // 3
 
 
def safe_trim(text, max_chars):
    return text[:max_chars]
 
 
def extract_wait_time(error_message):
    match = re.search(r'try again in ([0-9.]+)s', str(error_message))
    return float(match.group(1)) + 3.0 if match else 20.0
 
 
def groq_call(client, **kwargs):
    for attempt in range(10):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIStatusError) as e:
            wait = max(extract_wait_time(e), 13.0)
            print(f"[GROQ] Rate limit (attempt {attempt+1}/10). Waiting {wait:.1f}s...")
            time.sleep(wait)
    raise Exception("Groq API failed after all retries.")
 
 
def summarize(client, chunks, style="brief", query=""):
    from rag.citation_engine import chunks_to_plain_text
 
    print("=" * 60)
    print("SUMMARY AGENT  —  full PDF mode")
    print(f"  Style   : {style}")
    print(f"  Chunks  : {len(chunks)}  |  cap: {MAX_CHUNKS}")
    print(f"  Batch   : {CHUNKS_PER_BATCH} chunks/call  |  sleep: {SLEEP_BETWEEN}s")
    print("=" * 60)
 
    chunks = chunks[:MAX_CHUNKS]
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["detailed"])
    query_trimmed = safe_trim(query, 150)
 
    batches = [
        chunks[i : i + CHUNKS_PER_BATCH]
        for i in range(0, len(chunks), CHUNKS_PER_BATCH)
    ]
    print(f"  Total batches: {len(batches)}\n")
 
    all_summaries = []   # collect every batch result — NO merge bottleneck
 
    for idx, batch in enumerate(batches):
        print(f"[Batch {idx+1}/{len(batches)}]  {len(batch)} chunks")
 
        # ── build context: give each chunk fair share of char budget ──────
        chars_per_chunk = 2800 // max(len(batch), 1)
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
 
        total_est = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS
 
        # safety trim if over budget
        if total_est > TPM_HARD_LIMIT:
            allowed_chars = (TPM_HARD_LIMIT - OUTPUT_TOKENS - PROMPT_OVERHEAD) * 3
            context = safe_trim(context, allowed_chars)
            user_prompt = (
                f"User request: {query_trimmed}\n\n"
                f"--- PDF CONTENT ---\n{context}\n---\n\n"
                "Write complete exam notes for ALL topics in the content above."
            )
            total_est = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS
 
        print(f"  tokens ~{total_est}")
 
        if total_est > TPM_HARD_LIMIT:
            print("  [SKIP] still over budget")
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
 
        result = response.choices[0].message.content.strip()
        all_summaries.append(result)
        print(f"  [OK] {len(result)} chars returned")
 
        try:
            u = response.usage
            print(f"  [USAGE] prompt={u.prompt_tokens} completion={u.completion_tokens}")
        except Exception:
            pass
 
        if idx < len(batches) - 1:
            print(f"  sleeping {SLEEP_BETWEEN}s...\n")
            time.sleep(SLEEP_BETWEEN)
 
    if not all_summaries:
        return "No content could be summarised."
 
    print(f"\n✅ Done. {len(all_summaries)} sections collected.")
 
    # ── OUTPUT: just join all batch summaries with clear section dividers ──
    # No merge step = no bottleneck = full content preserved
    header = f"# Exam Notes — Full PDF Summary\n{'='*50}\n\n"
    divider = "\n\n" + "─" * 40 + "\n\n"
    return header + divider.join(all_summaries)