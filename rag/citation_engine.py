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
        elif hasattr(chunk, "text"):
            text = chunk.text
            source = getattr(chunk, "source", f"Source {i+1}")
        else:
            text = str(chunk)
            source = f"Source {i+1}"
        parts.append(f"[{i+1}] ({source})\n{text}")
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
            citation += f", page {page}"
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


# ── Groq-based citation answering ────────────────────────────


def answer_with_citations(client, query, chunks, chat_history=None):
    """Answer a query using chunks with citation support."""
    context = build_context_with_citations(chunks)
    context = context[:2000]  # hard cap

    history_text = "(no previous messages)"
    if chat_history:
        lines = []
        for msg in chat_history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:150]}")
        history_text = "\n".join(lines)

    system_prompt = (
        "You are a university study assistant. "
        "Answer using only the document context provided. "
        "Be clear and concise."
    )

    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"Document Context:\n{context}\n\n"
        f"Question: {query[:300]}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )

    return response.choices[0].message.content