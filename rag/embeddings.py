from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedder():
    return SentenceTransformer(_MODEL_NAME)


def embed_texts(embedder, texts):
    return embedder.encode(texts, show_progress_bar=False)


def embed_query(embedder, query):
    return embedder.encode([query])