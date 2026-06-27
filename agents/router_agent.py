import time

from rag.vector_store import search
from rag.citation_engine import (
    build_context_with_citations,
    format_citations_for_display,
    confidence_label,
)
from agents.quiz_agent import generate_quiz
from agents.summary_agent import summarize, detect_style
from agents.web_agent import answer_with_web_search

MODEL = "gemini-2.0-flash"


def gemini_call(client, system_prompt, user_prompt):
    """Call Gemini using the new google-genai SDK with retry logic."""
    for attempt in range(5):
        try:
            prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            wait = min(5 * (attempt + 1), 30)
            print(f"[GEMINI ERROR] attempt {attempt+1}: {type(e).__name__}: {e}")
            time.sleep(wait)
    raise Exception("Gemini API failed after all retries.")


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
    decision = gemini_call(client, "", prompt)
    decision = decision.strip().lower()

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

    retrieved = search(query, embedder, index, chunks, top_k=8)

    context = build_context_with_citations(retrieved)
    context = context[:5000]

    history_text = format_history(chat_history)

    chunk_count = len(retrieved) if retrieved else 0
    if chunk_count < 5:
        general_knowledge_instruction = (
            "PDF coverage is LOW. After answering from the PDF, "
            "add a clearly labeled '🌐 Additional Context (General Knowledge)' section "
            "with 3-5 relevant general knowledge points that EXTEND "
            "what the PDF says. Keep it directly relevant to the topic only. "
            "Never contradict the PDF content."
        )
    else:
        general_knowledge_instruction = (
            "PDF coverage is HIGH. Do NOT add any general knowledge. "
            "Answer strictly from the PDF context only."
        )

    system_prompt = """You are an expert university study assistant.
Use only the document context provided.
Answer in detailed numbered points suitable for a 10-15 mark exam answer.
Include definitions, explanations, examples and applications."""

    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"COVERAGE INSTRUCTION: {general_knowledge_instruction}"
    )

    answer = gemini_call(client, system_prompt, user_prompt)
    citations = format_citations_for_display(retrieved)
    confidence = confidence_label(retrieved)

    return answer, {"citations": citations, "confidence": confidence}


def _general_chat(client, query, chat_history):

    history_text = format_history(chat_history)
    system_prompt = "You are a friendly study assistant. Answer clearly and helpfully."
    user_prompt = f"Recent conversation:\n{history_text}\n\nMessage: {query}"

    return gemini_call(client, system_prompt, user_prompt)


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