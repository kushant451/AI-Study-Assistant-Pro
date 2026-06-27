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
            "continue", "elaborate", "expand", "tell more"
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

    if is_follow_up(query) and last_retrieved is not None:
        print(f"[FOLLOW-UP] Reusing cached chunks for: {last_topic}")
        retrieved = last_retrieved
        query = f"Expand further on: {last_topic}"
        print(f"[FOLLOW-UP] Using {len(retrieved)} cached chunks")
    else:
        retrieved = search(query, embedder, index, chunks, top_k=6)
        new_retrieved = retrieved
        new_topic = query
        print(f"[NEW QUERY] Searched {len(retrieved)} chunks for: {query[:50]}")

    context = build_context_with_citations(retrieved)
    context = context[:3000]
    history_text = format_history(chat_history)

    system_prompt = """You are an expert ICAI exam tutor.
Answer STRICTLY from the document context below only.
Do NOT add outside examples, companies, or theory not present in the text.
Do NOT invent limitations, advantages, or comparisons not in the context.
If asked for more detail, expand ONLY on the same topic as before.
Structure: definition → key points from text → conclusion."""

    user_prompt = f"""Context:
{context}

Conversation so far:
{history_text}

Question: {query[:300]}

Answer using ONLY the context above. If context is insufficient say: "The document does not cover this in detail." """

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