def build_context_with_citations(retrieved_chunks):
    parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.get("source", "unknown")
        parts.append(f"[Source {i+1} - {source}]\n{chunk['text']}")
    return "\n\n".join(parts)


def format_citations_for_display(retrieved_chunks, preview_length=400):
    citations = []
    for i, chunk in enumerate(retrieved_chunks):
        text = chunk["text"]
        preview = text[:preview_length] + ("..." if len(text) > preview_length else "")
        citations.append({
            "label": f"Source {i+1}",
            "source": chunk.get("source", "unknown"),
            "relevance": chunk["relevance"],
            "preview": preview,
        })
    return citations


def confidence_label(retrieved_chunks):
    if not retrieved_chunks:
        return "Low"

    best_relevance = max(c["relevance"] for c in retrieved_chunks)

    if best_relevance >= 70:
        return "High"
    elif best_relevance >= 40:
        return "Medium"
    else:
        return "Low"


def chunks_to_plain_text(chunks, limit=20):
    sample = chunks[:limit]
    return "\n\n".join(
        c["text"] if isinstance(c, dict) else c
        for c in sample
    )