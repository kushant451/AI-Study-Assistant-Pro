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

        # include page number in citation marker if available
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
    context = context[:3000]  # increased from 2000 for better coverage

    history_text = "(no previous messages)"
    if chat_history:
        lines = []
        for msg in chat_history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:150]}")
        history_text = "\n".join(lines)

    # ── UPDATED SYSTEM PROMPT ─────────────────────────────────
    system_prompt = """You are a helpful study assistant for university students.

Your job is to explain concepts from the PDF in simple, easy-to-understand language.

STRICT RULES:
1. Use ONLY the information from the Document Context provided below.
2. DO NOT copy-paste sentences directly from the document.
3. Explain in your OWN WORDS — like you are teaching a student simply.
4. Use simple analogies or examples if it helps understanding.
5. Stay 100% faithful to what the PDF says — do not add outside knowledge.
6. If the PDF content is insufficient to answer fully, say so honestly.
7. Always cite the page number at the end.

RESPONSE FORMAT:
📄 **Answer:**
[Your simple explanation in own words — faithful to PDF]

💡 **Example:** (only include if it genuinely helps understanding)
[A simple real-world analogy or example]

📍 **Source:** [mention page numbers from context markers above]
⚠️ **Note:** [only include if content in PDF is limited or partial]

IMPORTANT: If the question cannot be answered from the provided context, respond:
"❌ This topic was not found in the uploaded PDF. The document covers: [list main topics briefly]."
"""

    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"Document Context:\n{context}\n\n"
        f"Student Question: {query[:300]}\n\n"
        f"Remember: Explain simply in your own words. Do not copy from the document."
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,   # slightly higher than 0.1 for more natural rephrasing
        max_tokens=700,    # increased from 600 for complete answers
    )

    answer = response.choices[0].message.content

    # ── append coverage note below the LLM answer ─────────────
    note = coverage_note(chunks)
    if "⚠️" in note:
        answer += f"\n\n{note}"

    return answer