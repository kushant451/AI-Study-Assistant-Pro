def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents, chunk_size=1200, overlap=200):
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for c in chunks:
            all_chunks.append({"text": c, "source": doc["filename"]})
    return all_chunks


def get_document_stats(documents, chunks):
    combined_text = " ".join(doc["text"] for doc in documents)
    words = combined_text.split()
    word_count = len(words)
    reading_time_min = max(1, round(word_count / 200))

    return {
        "word_count": word_count,
        "char_count": len(combined_text),
        "chunk_count": len(chunks),
        "reading_time_min": reading_time_min,
        "document_count": len(documents),
        "filenames": [doc["filename"] for doc in documents],
    }