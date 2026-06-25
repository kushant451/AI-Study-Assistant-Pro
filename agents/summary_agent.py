import time
import re
from groq import RateLimitError

STYLE_PROMPTS = {
    "brief": "Summarize in 3-4 bullet points covering only key ideas.",
    "detailed": (
        "Create concise exam notes. Use numbered headings. "
        "3-5 bullet points each. Include definitions, pros/cons, uses. "
        "No repetition. Be brief but complete."
    )
}

TPM_HARD_LIMIT    = 6000
OUTPUT_TOKENS     = 180
PROMPT_OVERHEAD   = 120
MAX_CONTEXT_CHARS = 350


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


def safe_trim(text, max_tokens):
    return text[:max_tokens * 3]


def extract_wait_time(error_message):
    """Extract suggested wait time from Groq error message."""
    match = re.search(r'try again in ([0-9.]+)s', str(error_message))
    if match:
        return float(match.group(1)) + 1.0  # add 1s buffer
    return 10.0  # default fallback


def groq_call(client, **kwargs):
    for attempt in range(8):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = extract_wait_time(e)
            wait = max(wait, 3.0)  # always wait at least 3s
            print(f"[GROQ] Rate limit. Retry {attempt+1}/8 in {wait:.1f}s...")
            time.sleep(wait)
    raise Exception("Groq API rate limit exceeded after all retries.")


def build_and_validate_messages(system_prompt, user_prompt, max_tokens_out):
    total_est = count_tokens_approx(system_prompt + user_prompt) + max_tokens_out
    print(f"  [TOKEN CHECK] ~{total_est} / {TPM_HARD_LIMIT}")
    if total_est > TPM_HARD_LIMIT:
        raise ValueError(f"Too large: ~{total_est} tokens.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def summarize(client, chunks, style="brief", query=""):
    from rag.citation_engine import chunks_to_plain_text

    print("=" * 60)
    print("SUMMARY AGENT STARTED")
    print(f"  Style: {style} | Chunks: {len(chunks)}")
    print("=" * 60)

    system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["brief"])
    query_trimmed = query[:200]
    batch_summaries = []

    for idx, chunk in enumerate(chunks):
        print(f"\n[BATCH {idx+1}/{len(chunks)}]")
        time.sleep(12)  # 12s gap = max ~5 calls/min, safe under 6000 TPM

        context = chunks_to_plain_text([chunk], limit=1)
        context = safe_trim(context, max_tokens=250)
        user_prompt = f"Request: {query_trimmed}\n\nMaterial:\n{context}"

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

        batch_summaries.append(response.choices[0].message.content[:600])

    if not batch_summaries:
        return "No content could be summarised (all chunks were too large)."

    print("\n[MERGE STEP]")
    time.sleep(12)

    merge_token_budget = TPM_HARD_LIMIT - OUTPUT_TOKENS - PROMPT_OVERHEAD
    combined = ""
    for s in batch_summaries:
        candidate = combined + "\n\n" + s if combined else s
        if count_tokens_approx(candidate) <= merge_token_budget:
            combined = candidate
        else:
            print("  [MERGE] Budget reached — stopping.")
            break

    merge_prompt = (
        "Merge these summaries into clean exam-friendly study notes. "
        "Remove duplicates. Keep structure. Be concise.\n\n"
        f"CONTENT:\n{combined}"
    )

    try:
        messages = build_and_validate_messages(system_prompt, merge_prompt, OUTPUT_TOKENS)
    except ValueError as e:
        print(f"  [MERGE FALLBACK] {e}")
        return combined

    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        max_tokens=OUTPUT_TOKENS,
    )

    print("\nSUMMARY COMPLETE")
    return response.choices[0].message.content