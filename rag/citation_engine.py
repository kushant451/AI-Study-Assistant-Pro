import time

# ── Citation engine ───────────────────────────────────────────


def chunks_to_plain_text(chunks, limit=5):
    """Convert chunks to plain text for summarization."""
    texts = []
    for chunk in chunks[:limit]:
        if isinstance(chunk, dict):
            texts.append(chunk.get("text", ""))
        elif hasattr(chunk, "text"):
            texts.append(chunk.text)
        else:
            texts.append(str(chunk))
    return "\n\n".join(texts)


def build_context_with_citations(chunks):
    """Build context string with citation markers."""
    parts = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            source = chunk.get("source", f"Source {i+1}")
            page = chunk.get("page", None)
        elif hasattr(chunk, "text"):
            text = chunk.text
            source = getattr(chunk, "source", f"Source {i+1}")
            page = getattr(chunk, "page", None)
        else:
            text = str(chunk)
            source = f"Source {i+1}"
            page = None

        label = f"[{i+1}]"
        if page:
            label += f" (Page {page})"
        parts.append(f"{label}\n{text}")
    return "\n\n".join(parts)


def format_citations_for_display(chunks):
    """Format citations for display in the UI."""
    citations = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            source = chunk.get("source", f"Source {i+1}")
            page = chunk.get("page", None)
        elif hasattr(chunk, "source"):
            source = chunk.source
            page = getattr(chunk, "page", None)
        else:
            source = f"Source {i+1}"
            page = None

        citation = f"[{i+1}] {source}"
        if page:
            citation += f", Page {page}"
        citations.append(citation)
    return citations


def confidence_label(chunks):
    """Return a confidence label based on number of retrieved chunks."""
    if not chunks:
        return "Low"
    if len(chunks) >= 5:
        return "High"
    if len(chunks) >= 3:
        return "Medium"
    return "Low"


def coverage_note(chunks):
    """Return a coverage note based on number of chunks retrieved."""
    count = len(chunks) if chunks else 0
    if count == 0:
        return "⚠️ No relevant content found in the uploaded PDF."
    if count >= 5:
        return "✅ Good coverage found in the PDF."
    if count >= 3:
        return "⚠️ Moderate coverage — PDF has limited content on this topic."
    return "⚠️ Very limited coverage — PDF briefly mentions this topic."


# ── Groq-based citation answering ────────────────────────────


def answer_with_citations(client, query, chunks, chat_history=None):
    """Answer a query using chunks with citation support."""
    context = build_context_with_citations(chunks)
    context = context[:5000]  # increased for better coverage

    history_text = "(no previous messages)"
    if chat_history:
        lines = []
        for msg in chat_history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:150]}")
        history_text = "\n".join(lines)

    # ── coverage check → decide if general knowledge needed ───
    chunk_count = len(chunks) if chunks else 0
    if chunk_count < 5:
        general_knowledge_instruction = (
            "PDF coverage is LOW. After answering from the PDF, "
            "add a clearly labeled '🌐 Additional Context (General Knowledge)' section "
            "with 3-5 relevant general knowledge points that EXTEND "
            "what the PDF says. Keep it directly relevant to the topic only. "
            "Never contradict the PDF content."
        )
    else:
        general_knowledge_instruction = (
            "PDF coverage is HIGH. Do NOT add any general knowledge. "
            "Answer strictly from the PDF context only."
        )

    # ── SYSTEM PROMPT ─────────────────────────────────────────
    system_prompt = """You are a strict document-based study assistant for university students.

ABSOLUTE RULES:
1. Use ONLY the information from the Document Context provided.
2. DO NOT copy-paste sentences directly from the document.
3. Explain in simple own words but ONLY what the document says.
4. Stay 100% faithful to what the PDF says — no invented facts.
5. DO NOT add outside knowledge in the main PDF answer section.
6. Do NOT include any dates or years unless the PDF explicitly states them.
7. Do NOT mention cloud computing, AI, blockchain unless in the PDF context.

RESPONSE FORMAT:

📄 **From Your PDF:**
[Answer strictly from PDF context in simple, clear words]

📍 **Source:** [mention page numbers from context markers like (Page X)]

---

🌐 **Additional Context (General Knowledge):**
[ONLY include this section when PDF coverage is LOW.
Add 3-5 relevant general knowledge points that extend the PDF.
Clearly label as general knowledge.
Must be directly relevant to the topic — no off-topic content.
Never contradict what the PDF says.]

⚠️ **Note:** [only if PDF content is limited or partial]

IMPORTANT:
- If question is about Evolution of ERP:
  PDF only LISTS system names MIS, IIS, EIS, CIS, EWS without descriptions.
  Do NOT add descriptions for these — just say they were stepping stones.
  Only describe MRP, MRPII and ERP in detail as PDF explains those.
- If topic is not found in context at all, respond:
  '❌ This topic was not found in the uploaded PDF.'
"""

    # ── USER PROMPT ───────────────────────────────────────────
    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"DOCUMENT CONTEXT (answer ONLY from this):\n{context}\n\n"
        f"Student Question: {query[:300]}\n\n"
        f"STRICT INSTRUCTION: Only use the above context. "
        f"The context does not mention any dates or years — "
        f"so do NOT include any dates in your answer. "
        f"Do NOT mention cloud computing, AI, or blockchain "
        f"unless they appear in the context above. "
        f"If a fact is not in the context, do not say it.\n\n"
        f"CRITICAL: If the question is about Evolution of ERP, "
        f"the PDF only LISTS system names like MIS, IIS, EIS, CIS, EWS "
        f"without any descriptions. Do NOT add descriptions for these — "
        f"just mention they were introduced as stepping stones. "
        f"Only describe MRP, MRPII and ERP in detail as the PDF explains those.\n\n"
        f"COVERAGE INSTRUCTION: {general_knowledge_instruction}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,   # increased for general knowledge section
    )

    answer = response.choices[0].message.content

    # ── append coverage note below the LLM answer ─────────────
    note = coverage_note(chunks)
    if "⚠️" in note:
        answer += f"\n\n{note}"

    return answer