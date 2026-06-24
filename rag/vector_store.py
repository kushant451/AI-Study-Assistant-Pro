import numpy as np
import faiss

from rag.embeddings import embed_texts, embed_query


def build_vector_store(chunks, embedder):
    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(embedder, texts)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def search(query, embedder, index, chunks, top_k=15):
    print("QUERY RECEIVED:", query)

    if index is None:
        raise ValueError(
            "FAISS index is None. Process PDFs first."
        )

    query_vector = np.array(
        embed_query(embedder, query),
        dtype="float32"
    )

    print("Original query shape:", query_vector.shape)

    
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)

    print("Final query shape:", query_vector.shape)

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for dist, idx in zip(
        distances[0],
        indices[0]
    ):
        if 0 <= idx < len(chunks):

            relevance = max(
                0,
                100 - (dist * 10)
            )

            results.append(
                {
                    "text": chunks[idx]["text"],
                    "source": chunks[idx].get(
                        "source",
                        "unknown"
                    ),
                    "chunk_id": int(idx),
                    "relevance": round(
                        float(relevance),
                        1
                    ),
                }
            )

    print("\n===== RETRIEVED CHUNKS =====")

    for i, r in enumerate(results):
        print(f"\nChunk {i+1}")
        print(r["text"][:1000])

    print("\n============================")

    return results