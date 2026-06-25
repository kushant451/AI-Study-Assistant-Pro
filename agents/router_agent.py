from rag.vector_store import search
from rag.citation_engine import (
    build_context_with_citations,
    format_citations_for_display,
    confidence_label,
)
from agents.quiz_agent import generate_quiz
from agents.summary_agent import summarize, detect_style
from agents.web_agent import answer_with_web_search


def is_follow_up(query: str):
    return any(
        k in query.lower() for k in [
            "more theory", "more details", "explain more",
            "continue", "elaborate", "expand", "tell more"
        ]
    )


def is_exam_question(query: str):
    q = query.lower()
    return any(
        k in q for k in [
            "explain", "describe", "in detail", "elaborate",
            "write note", "long answer", "10 marks", "15 marks", "evolution"
        ]
    )


ROUTER_PROMPT = """You are a routing assistant. Reply with ONLY one word.

Tools: doc_qa, summarize, quiz, web_search, general_chat

Recent conversation:
{history}

User message: "{query}"

Answer with exactly one word."""


def format_history(chat_history):
    if not chat_history:
        return "(no previous messages)"
    lines = []
    for msg in chat_history[-3:]:  # reduced from 6 to 3
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:150]  # reduced from 400 to 150
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def route_query(client, query, has_documents, chat_history):

    history_text = format_history(chat_history)
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

    if not has_documents:
        prompt = (
            "Reply with ONLY one word: 'web_search' or 'general_chat'.\n\n"
            f"Question: {query[:200]}"
        )
    else:
        prompt = ROUTER_PROMPT.format(history=history_text, query=query[:200])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,  # only needs one word
    )

    decision = response.choices[0].message.content.strip().lower()
    valid_tools = ["doc_qa", "summarize", "quiz", "web_search", "general_chat"]
    for tool in valid_tools:
        if tool in decision:
            return tool
    return "general_chat"


def _doc_qa(client, query, embedder, index, chunks, chat_history):

    if is_follow_up(query):
        for msg in reversed(chat_history):
            if msg["role"] == "user" and not is_follow_up(msg["content"]):
                query = (
                    f"Expand each point in detail about {msg['content']} "
                    f"using only the document context."
                )
                break

    retrieved = search(query, embedder, index, chunks, top_k=5)  # reduced from 8
    print("\n====================")
    print("QUERY:", query)
    print("====================")

    context = build_context_with_citations(retrieved)
    context = context[:1500]  # hard cap context

    history_text = format_history(chat_history)

    system_prompt = (
        "You are a university study assistant. "
        "Use only the document context. "
        "Answer in clear numbered points suitable for a 10-mark exam answer."
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
        max_tokens=800,  # reduced from 2000
    )

    answer = response.choices[0].message.content
    citations = format_citations_for_display(retrieved)
    confidence = confidence_label(retrieved)

    return answer, {"citations": citations, "confidence": confidence}


def _general_chat(client, query, chat_history):

    history_text = format_history(chat_history)
    system_prompt = "You are a friendly study assistant. Answer clearly and concisely."
    user_prompt = f"Recent conversation:\n{history_text}\n\nMessage: {query[:300]}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,  # reduced from 2000
    )

    return response.choices[0].message.content


def run_agent(client, query, embedder=None, index=None, chunks=None, chat_history=None):

    chat_history = chat_history or []
    has_documents = chunks is not None and len(chunks) > 0

    tool = route_query(client, query, has_documents, chat_history)

    extra = None
    history_text = format_history(chat_history)

    if tool == "doc_qa":
        answer, extra = _doc_qa(client, query, embedder, index, chunks, chat_history)

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

    return tool, answer, extra