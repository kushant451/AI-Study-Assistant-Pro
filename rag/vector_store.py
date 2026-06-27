import numpy as np
import faiss

from rag.embeddings import embed_texts, embed_query

# ── Query expansion map ───────────────────────────────────────
# When user asks vague questions, expand to richer search terms
# so FAISS finds the RIGHT chunks from the PDF

QUERY_EXPANSIONS = {
    # ── follow-up fallbacks (overridden by last_topic in router) ──
    "more theory": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "add more theory": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "more details": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "explain more": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "elaborate": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "expand": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "tell more": "ERP MRP MRPII evolution enterprise resource planning integrated business",
    "continue": "ERP MRP MRPII evolution enterprise resource planning integrated business",

    # ── existing entries ──
    "evolution of erp": "evolution ERP MRP MRPII history development MPS BOM material requirement planning manufacturing",
    "evolution": "evolution history development MRP MRPII ERP growth timeline",
    "what is erp": "ERP definition enterprise resource planning integrated business management system database",
    "erp definition": "ERP definition enterprise resource planning integrated business management system",
    "benefits of erp": "ERP benefits advantages improved efficiency reduced costs productivity customer service",
    "features of erp": "ERP features multi-platform supply chain integrated information system workflow",
    "erp characteristics": "ERP characteristics flexible modular comprehensive open architecture best practices",
    "bpr": "BPR business process reengineering radical redesign fundamental rethinking dramatic improvement",
    "what is bpr": "BPR business process reengineering Hammer Champhy radical redesign fundamental",
    "erp implementation": "ERP implementation steps methodology consultants customization gap analysis",
    "implementation steps": "ERP implementation steps identifying needs as is would be reengineering packages",
    "erp failure": "ERP failure reasons people resistance customization political fights",
    "why erp fails": "ERP failure reasons resistance change customization political",
    "post implementation": "post implementation ERP expectations fears KPI CSF blues problems",
    "risk": "ERP risk governance single point failure structural changes job role online realtime",
    "erp vendors": "ERP vendors SAP Baan Oracle BPCS MFG Pro R3 system 21",
    "sap": "SAP modules financials controlling investment treasury sales distribution production HR",
    "sap modules": "SAP R3 modules financials controlling investment treasury sales distribution production materials HR",
    "mrp": "MRP material requirement planning master production schedule BOM bill of material planned orders",
    "mrpii": "MRPII manufacturing requirement planning operational financial planning simulation what if",
    "three tier": "three tier architecture client server presentation application data storage RDBMS",
    "architecture": "ERP architecture three tier client server RDBMS enabling technologies",
    "enabling technologies": "ERP enabling technologies EDI workflow groupware internet intranet data warehousing",
    "ecommerce": "ERP ecommerce integration web internet customers suppliers business to business",
    "business engineering": "business engineering IT information technology BPR process oriented value chain",
    "business modelling": "business modelling SAP EPC event driven process chain flowchart MIS planning",
    "why companies use erp": "companies undertake ERP financial integration customer order standardize manufacturing HR",
    "hr": "human resources HR personnel recruitment travel benefits payroll time management SAP",
    "financials": "SAP financials general ledger accounts receivable payable fixed assets financial accounting",
    "materials management": "materials management purchasing inventory warehouse invoice verification MM",
    "production planning": "production planning MRP capacity SOP demand management KANBAN repetitive manufacturing",
    "videocon": "Videocon case study ERP SAP implementation India factory",
    "case study": "case study Videocon AirTouch ERP implementation real world example",
}


def expand_query(query: str) -> str:
    """
    Expand a short/vague query into richer search terms
    so FAISS retrieves the most relevant chunks.
    """
    q_lower = query.lower().strip()

    # exact match first
    if q_lower in QUERY_EXPANSIONS:
        expanded = QUERY_EXPANSIONS[q_lower]
        print(f"[QUERY EXPANSION] '{query}' → '{expanded}'")
        return expanded

    # partial match — check if any key is contained in query
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in q_lower or q_lower in key:
            expanded = f"{query} {expansion}"
            print(f"[QUERY EXPANSION] '{query}' → '{expanded}'")
            return expanded

    # no expansion found — use original
    print(f"[QUERY EXPANSION] No expansion for '{query}' — using original")
    return query


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

    # ── expand query before embedding ────────────────────────
    expanded_query = expand_query(query)

    query_vector = np.array(
        embed_query(embedder, expanded_query),
        dtype="float32"
    )

    print("Query vector shape:", query_vector.shape)

    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)

    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks):

            relevance = max(0, 100 - (dist * 10))

            chunk = chunks[idx]
            results.append(
                {
                    "text":      chunk["text"],
                    "source":    chunk.get("source", "unknown"),
                    "page":      chunk.get("page", None),
                    "chunk_id":  int(idx),
                    "relevance": round(float(relevance), 1),
                    "distance":  round(float(dist), 4),
                }
            )

    # ── sort by relevance (closest distance first) ────────────
    results.sort(key=lambda x: x["distance"])

    print("\n===== RETRIEVED CHUNKS =====")
    for i, r in enumerate(results):
        page_info = f"Page {r['page']}" if r.get("page") else "Page ?"
        print(f"\nChunk {i+1} | {page_info} | Relevance: {r['relevance']} | Distance: {r['distance']}")
        print(r["text"][:300])
    print("\n============================")

    return results