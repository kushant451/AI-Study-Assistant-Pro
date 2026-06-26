import time
import re
from groq import RateLimitError, APIStatusError

STYLE_PROMPTS = {
    "brief": "Summarize in 3-4 bullet points covering only key ideas.",
    "detailed": (
        "Create concise exam notes. Use numbered headings. "
        "3-5 bullet points each. Include definitions, pros/cons, uses. "
        "No repetition. Be brief but complete."
    )
}

TPM_HARD_LIMIT  = 6000
OUTPUT_TOKENS   = 150
PROMPT_OVERHEAD = 100
MAX_CHUNKS      = 5


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
            # catches both 429 RateLimitError and 413 APIStatusError
            wait = max(extract_wait_time(e), 8.0)
            print(f"[GROQ] Error caught (attempt {attempt+1}/8). Waiting {wait:.1f}s... Error: {e}")
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

    for idx, chunk in enumerate(chunks):
        print(f"\n[BATCH {idx+1}/{len(chunks)}]")
        time.sleep(10)

        context = chunks_to_plain_text([chunk], limit=1)
        context = safe_trim(context, 200)

        user_prompt = f"Request: {query_trimmed}\n\nMaterial:\n{context}"

        total_est = count_tokens_approx(system_prompt + user_prompt) + OUTPUT_TOKENS
        print(f"  [TOKEN CHECK] ~{total_est} / {TPM_HARD_LIMIT}")

        if total_est > TPM_HARD_LIMIT:
            print(f"  [SKIP] Too large: {total_est} tokens")
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

        batch_summaries.append(response.choices[0].message.content[:400])

    if not batch_summaries:
        return "No content could be summarised."

    print("\n[MERGE STEP]")
    time.sleep(10)

    combined = "\n\n".join(batch_summaries)
    combined = safe_trim(combined, 1500)

    merge_prompt = (
        "Merge into clean exam-friendly study notes. "
        "Remove duplicates. Keep structure. Be concise.\n\n"
        f"CONTENT:\n{combined}"
    )

    total_est = count_tokens_approx(system_prompt + merge_prompt) + OUTPUT_TOKENS
    print(f"  [MERGE TOKEN CHECK] ~{total_est} / {TPM_HARD_LIMIT}")

    if total_est > TPM_HARD_LIMIT:
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