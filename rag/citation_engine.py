import time

MODEL = "gemini-2.0-flash"


def chunks_to_plain_text(chunks, limit=5):
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
    if not chunks:
        return "Low"
    if len(chunks) >= 5:
        return "High"
    if len(chunks) >= 3:
        return "Medium"
    return "Low"


def coverage_note(chunks):
    count = len(chunks) if chunks else 0
    if count == 0:
        return "⚠️ No relevant content found in the uploaded PDF."
    if count >= 5:
        return "✅ Good coverage found in the PDF."
    if count >= 3:
        return "⚠️ Moderate coverage — PDF has limited content on this topic."
    return "⚠️ Very limited coverage — PDF briefly mentions this topic."


def answer_with_citations(client, query, chunks, chat_history=None):
    """Answer a question using document chunks. Uses new google-genai SDK."""
    context = build_context_with_citations(chunks)
    context = context[:5000]

    history_text = "(no previous messages)"
    if chat_history:
        lines = []
        for msg in chat_history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:150]}")
        history_text = "\n".join(lines)

    chunk_count = len(chunks) if chunks else 0
    if chunk_count < 5:
        general_knowledge_instruction = (
            "PDF coverage is LOW. After answering from the PDF, "
            "add a clearly labeled '🌐 Additional Context (General Knowledge)' section "
            "with 3-5 relevant general knowledge points. "
            "Never contradict the PDF content."
        )
    else:
        general_knowledge_instruction = (
            "PDF coverage is HIGH. Answer strictly from PDF context only."
        )

    system_prompt = """You are a strict document-based study assistant for university students.
Use ONLY the information from the Document Context provided.
Answer in detailed numbered points suitable for a 10-15 mark exam answer."""

    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"Question: {query[:300]}\n\n"
        f"COVERAGE INSTRUCTION: {general_knowledge_instruction}"
    )

    prompt = f"{system_prompt}\n\n{user_prompt}"

    # ✅ New google-genai SDK call
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            answer = response.text
            break
        except Exception as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"[GEMINI ERROR] attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(wait)
    else:
        raise Exception("Gemini API failed after all retries.")

    note = coverage_note(chunks)
    if "⚠️" in note:
        answer += f"\n\n{note}"

    return answer