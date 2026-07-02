import re
import time
from groq import RateLimitError, APIStatusError

from rag.vector_store import search
from rag.citation_engine import (
    build_context_with_citations,
    format_citations_for_display,
    confidence_label,
)
from agents.quiz_agent import generate_quiz
from agents.summary_agent import summarize, detect_style
from agents.web_agent import answer_with_web_search


def extract_wait_time(error_message):
    match = re.search(r'try again in ([0-9.]+)s', str(error_message))
    return float(match.group(1)) + 2.0 if match else 15.0


def groq_call(client, **kwargs):
    for attempt in range(8):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIStatusError) as e:
            wait = max(extract_wait_time(e), 8.0)
            print(f"[GROQ] Error (attempt {attempt+1}/8). Waiting {wait:.1f}s...")
            time.sleep(wait)
    raise Exception("Groq API failed after all retries.")


def is_follow_up(query: str):
    return any(
        k in query.lower() for k in [
            "more theory", "more details", "explain more",
            "continue", "elaborate", "expand", "tell more",
            "add more"
        ]
    )


def format_history(chat_history):
    if not chat_history:
        return "(no previous messages)"
    lines = []
    for msg in chat_history[-3:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:150]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


BPR_KEYWORDS = [
    "reengineering", "bpr", "radical redesign", "cleanslate",
    "clean slate", "tqm", "total quality", "suggestion scheme",
    "six sigma", "lean management", "hammer and champy",
    "fundamental rethinking", "dramatic improvement"
]

ERP_KEYWORDS = [
    "erp", "mrp", "mrpii", "mrp iii", "enterprise resource", "material requirement",
    "manufacturing resource", "money resource", "bill of material", "bom", 
    "master production", "evolution of erp", "evolution erp"
]


def is_bpr_chunk(chunk: dict) -> bool:
    text = chunk.get("text", "").lower()
    bpr_hits = sum(1 for k in BPR_KEYWORDS if k in text)
    erp_hits = sum(1 for k in ERP_KEYWORDS if k in text)
    return bpr_hits > erp_hits


def filter_chunks_by_topic(retrieved: list, topic: str) -> list:
    topic_lower = topic.lower()
    is_erp_query = any(k in topic_lower for k in ERP_KEYWORDS)
    is_bpr_query = any(k in topic_lower for k in BPR_KEYWORDS)

    if is_erp_query and not is_bpr_query:
        filtered = [c for c in retrieved if not is_bpr_chunk(c)]
        print(f"[FILTER] ERP topic: removed {len(retrieved)-len(filtered)} BPR chunks, kept {len(filtered)}")
        return filtered if filtered else retrieved

    if is_bpr_query and not is_erp_query:
        filtered = [c for c in retrieved if is_bpr_chunk(c)]
        print(f"[FILTER] BPR topic: kept {len(filtered)} BPR chunks")
        return filtered if filtered else retrieved

    return retrieved


def route_query(client, query, has_documents, chat_history):
    q = query.lower()

    if has_documents:
        if "quiz" in q or "mcq" in q:
            return "quiz"
        if any(x in q for x in [
            "summary", "summarize", "full pdf", "complete pdf",
            "entire pdf", "whole pdf", "full notes", "complete notes"
        ]):
            return "summarize"
        return "doc_qa"

    prompt = (
        "Reply with ONLY one word: 'web_search' or 'general_chat'.\n\n"
        f"Question: {query[:200]}"
    )

    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    decision = response.choices[0].message.content.strip().lower()
    valid_tools = ["doc_qa", "summarize", "quiz", "web_search", "general_chat"]
    for tool in valid_tools:
        if tool in decision:
            return tool
    return "general_chat"


def _doc_qa(client, query, embedder, index, chunks, chat_history,
            last_retrieved=None, last_topic=None):

    new_retrieved = last_retrieved
    new_topic = last_topic
    effective_topic = last_topic if (is_follow_up(query) and last_topic) else query

    if is_follow_up(query) and last_retrieved is not None:
        print(f"[FOLLOW-UP] Searching with original topic: '{last_topic}'")
        retrieved = search(last_topic, embedder, index, chunks, top_k=8)
        retrieved = filter_chunks_by_topic(retrieved, last_topic)
        query = f"Provide more detailed explanation about '{last_topic}' using only the document."
        new_retrieved = retrieved      # ← update cache with fresh filtered chunks
        new_topic = last_topic         # ← keep topic locked, don't reset
    else:
        retrieved = search(query, embedder, index, chunks, top_k=8)
        retrieved = filter_chunks_by_topic(retrieved, query)
        new_retrieved = retrieved
        new_topic = query
        print(f"[NEW QUERY] {len(retrieved)} chunks after filter for: {query[:50]}")

    context = build_context_with_citations(retrieved)
    context = context[:6000]
    history_text = format_history(chat_history)

    system_prompt = f"""You are an expert ICAI exam tutor.
CURRENT TOPIC: '{effective_topic}'
You MUST answer ONLY about the CURRENT TOPIC above.
STRICTLY FORBIDDEN: Do not mention BPR, Business Process Reengineering, TQM, Six Sigma, Lean, or ANY topic other than '{effective_topic}'.
Use ONLY context sentences that directly discuss '{effective_topic}'.
Ignore all other context even if present.
Do NOT add outside examples, theory, dates, years, decades, or statistics not explicitly present in the context.
If the context does not mention specific years or a timeline, do NOT invent one.
Do NOT write in paragraph format.
Use this structure with headings and bullet points:
## Definition
## Key Points
## Evolution Chain
## Conclusion"""

    user_prompt = f"""Context:
{context}

Conversation so far:
{history_text}

Question: {query[:300]}

Answer using ONLY the context above about '{effective_topic}'.
If the context mentions MRP, MRPII, or MRP III (Money Resource Planning), include ALL of them in the evolution chain.
If context is insufficient say: "The document does not cover this in detail." """

    response = groq_call(
        client,
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer = response.choices[0].message.content
    citations = format_citations_for_display(retrieved)
    confidence = confidence_label(retrieved)

    return answer, {"citations": citations, "confidence": confidence}, new_retrieved, new_topic


def _general_chat(client, query, chat_history):

    history_text = format_history(chat_history)
    system_prompt = "You are a friendly study assistant. Answer clearly and concisely."
    user_prompt = f"Recent conversation:\n{history_text}\n\nMessage: {query[:300]}"

    response = groq_call(
        client,
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content


def run_agent(client, query, embedder=None, index=None, chunks=None,
              chat_history=None, last_retrieved=None, last_topic=None):

    chat_history = chat_history or []
    has_documents = chunks is not None and len(chunks) > 0
    tool = route_query(client, query, has_documents, chat_history)

    extra = None
    new_retrieved = last_retrieved
    new_topic = last_topic
    history_text = format_history(chat_history)

    if tool == "doc_qa":
        answer, extra, new_retrieved, new_topic = _doc_qa(
            client, query, embedder, index, chunks, chat_history,
            last_retrieved=last_retrieved, last_topic=last_topic
        )
    elif tool == "summarize":
        style = detect_style(query)
        answer = summarize(client, chunks, style=style, query=query)
    elif tool == "quiz":
        extra = generate_quiz(client, chunks)
        answer = "Here's a quiz based on your material."
    elif tool == "web_search":
        answer = answer_with_web_search(client, query, history_text)
    else:
        answer = _general_chat(client, query, chat_history)

    return tool, answer, extra, new_retrieved, new_topic