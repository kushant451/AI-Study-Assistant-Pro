import time
import re
from groq import RateLimitError, APIStatusError

STYLE_PROMPTS = {
    "brief": "Summarize in 5-6 bullet points covering only key ideas. Be concise.",
    "detailed": (
        "Create detailed exam notes. Use numbered headings and sub-headings. "
        "5-8 bullet points per section. Include definitions, pros/cons, examples, uses. "
        "No repetition. Cover everything important."
    )
}

TPM_HARD_LIMIT   = 5500   # safely under Groq's 6000 TPM limit
OUTPUT_TOKENS    = 800    # was 150 — enough for real notes per batch
PROMPT_OVERHEAD  = 200
MAX_CHUNKS       = 117    # process ALL chunks — no artificial cap
CHUNKS_PER_BATCH = 8      # 8 chunks per call = ~15 calls for 117 chunks
SLEEP_BETWEEN    = 12     # seconds between API calls (safe for Groq free tier)


def detect_style(query):
    query_lower = query.lower()
    detailed_keywords = [
        "full pdf", "entire pdf", "whole pdf", "complete pdf",
        "full summary", "complete summary", "entire summary",
        "detail", "detailed", "in depth", "elaborate",
        "more summary", "expand", "long summary",
        "exam", "revision", "important points", "key points", "notes"
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
            wait = max(extract_wait_time(e), 12.0)
            print(f"[GROQ] Rate limit (attempt {attempt+1}/10). Waiting {wait:.1f}s...")
            time.sleep(wait)
    raise Exception("Groq API failed after all retries.")


def build_batch_prompt(chunks, query_trimmed, chunks_to_plain_text):
    """Combine multiple chunks into one context block, trimmed to fit token budget."""
    combined = ""
    chars_per_chunk = 2400 // max(len(chunks), 1)
    for chunk in chunks:
        piece = chunks_to_plain_text([chunk], limit=1)
        combined += safe_trim(piece, chars_per_chunk) + "\n\n"
    return f"Request: {query_trimmed}\n\nMaterial:\n{combined.strip()}"


def summarize(client, chunks, style="brief", query=""):
    from rag.citation_engine import chunks_to_plain_text

    print("=" * 60)
    print("SUMMARY AGENT STARTED  (full-PDF mode)")
    print(f"  Style   : {style}")
    print(f"  Chunks  : {len(chunks)} total  |  cap: {MAX_CHUNKS}")
    print(f"  Batch sz: {CHUNKS_PER_BATCH}  |  sleep: {SLEEP_BETWEEN}s")
    print("=" * 60)

    chunks = chunks[:MAX_CHUNKS]
    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])
    query_trimmed = query[:150]

    # ── PASS 1: summarize every batch of chunks ────────────────────────────
    batches = [
        chunks[i : i + CHUNKS_PER_BATCH]
        for i in range(0, len(chunks), CHUNKS_PER_BATCH)
    ]
    print(f"\n[PASS 1]  {len(batches)} batches to process...")

    pass1_summaries = []
    for idx, batch in enumerate(batches):
        print(f"\n  Batch {idx+1}/{len(batches)}  ({len(batch)} chunks)")

        user_prompt = build_batch_prompt(batch, query_trimmed, chunks_to_plain_text)
        total_est   = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS

        if total_est > TPM_HARD_LIMIT:
            user_prompt = safe_trim(user_prompt, (TPM_HARD_LIMIT - OUTPUT_TOKENS - PROMPT_OVERHEAD) * 3)
            total_est   = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS

        print(f"  [TOKENS] ~{total_est}")

        if total_est > TPM_HARD_LIMIT:
            print(f"  [SKIP] Cannot fit in token budget.")
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

        summary_text = response.choices[0].message.content
        pass1_summaries.append(summary_text)
        print(f"  [OK] Got {len(summary_text)} chars")

        try:
            print(f"  [USAGE] {response.usage}")
        except Exception:
            pass

        if idx < len(batches) - 1:
            print(f"  [SLEEP] {SLEEP_BETWEEN}s ...")
            time.sleep(SLEEP_BETWEEN)

    if not pass1_summaries:
        return "No content could be summarised."

    print(f"\n[PASS 1 DONE]  {len(pass1_summaries)} summaries collected.")

    # ── PASS 2: merge all summaries → final notes ─────────────────────────
    def merge_group(summaries, label="MERGE"):
        combined = "\n\n---\n\n".join(summaries)
        combined = safe_trim(combined, 3500)
        merge_system = (
            "You are an expert study notes editor. "
            "Merge the notes below into clean, structured exam-friendly notes. "
            "Use numbered headings and sub-headings. "
            "Remove duplicates. Keep ALL unique key points. "
            "Be thorough — this is for exam revision."
        )
        merge_prompt = f"Merge these notes into final exam notes:\n\n{combined}"
        total_est = count_tokens_approx(merge_system + merge_prompt) + OUTPUT_TOKENS

        print(f"\n[{label}] ~{total_est} tokens")
        time.sleep(SLEEP_BETWEEN)

        if total_est > TPM_HARD_LIMIT:
            combined = safe_trim(combined, 2000)
            merge_prompt = f"Merge these notes into final exam notes:\n\n{combined}"

        response = groq_call(
            client,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": merge_system},
                {"role": "user",   "content": merge_prompt},
            ],
            temperature=0.3,
            max_tokens=OUTPUT_TOKENS,
        )
        return response.choices[0].message.content

    GROUP_SIZE = 5
    if len(pass1_summaries) <= GROUP_SIZE:
        final = merge_group(pass1_summaries, label="FINAL MERGE")
    else:
        # Large PDF: mini-merges first, then final merge
        print(f"\n[PASS 2]  Mini-merging {len(pass1_summaries)} summaries in groups of {GROUP_SIZE}...")
        mini_merged = []
        for i in range(0, len(pass1_summaries), GROUP_SIZE):
            group = pass1_summaries[i : i + GROUP_SIZE]
            result = merge_group(group, label=f"MINI-MERGE {i//GROUP_SIZE + 1}")
            mini_merged.append(result)

        print(f"\n[PASS 2 DONE]  {len(mini_merged)} mini-summaries → final merge")
        final = merge_group(mini_merged, label="FINAL MERGE")

    print("\nSUMMARY COMPLETE")
    return final