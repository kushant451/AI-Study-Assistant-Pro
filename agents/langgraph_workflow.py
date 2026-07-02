from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

from rag.vector_store import search
from rag.citation_engine import (
    build_context_with_citations,
    format_citations_for_display,
    confidence_label,
)
from agents.quiz_agent import generate_quiz
from agents.summary_agent import summarize, detect_style
from agents.web_agent import answer_with_web_search
from agents.router_agent import format_history, route_query, groq_call


class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    client: Any
    embedder: Any
    index: Any
    chunks: Optional[List[str]]
    tool: Optional[str]
    answer: Optional[str]
    extra: Optional[Any]


def router_node(state: AgentState) -> dict:
    has_documents = state["chunks"] is not None and len(state["chunks"]) > 0
    tool = route_query(
        state["client"],
        state["query"],
        has_documents,
        state["chat_history"],
    )
    return {"tool": tool}


def doc_qa_node(state: AgentState) -> dict:
    retrieved = search(
        state["query"],
        state["embedder"],
        state["index"],
        state["chunks"],
        top_k=8,
    )
    context = build_context_with_citations(retrieved)
    context = context[:6000]
    history_text = format_history(state["chat_history"])

    system_prompt = """You are an expert ICAI exam tutor.
    Answer ONLY using the context provided below.
    STRICTLY FORBIDDEN: Do NOT add any dates, years, decades, statistics, timelines, or examples that are not explicitly present in the context.
    Do NOT use your own training knowledge to fill gaps.
    ONLY if the context contains NO relevant information at all, say: "The document does not cover this in detail." Do NOT add this line if you have already answered the question.
    Use headings and bullet points for exam-oriented explanations."""
    user_prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Current Question: {state['query']}\n\n"
        "REMINDER: Answer ONLY from the context above. Do NOT invent dates, timelines, or facts not in the context."
    )

    response = groq_call(
        state["client"],
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )

    answer = response.choices[0].message.content
    extra = {
        "citations": format_citations_for_display(retrieved),
        "confidence": confidence_label(retrieved),
    }

    return {"answer": answer, "extra": extra}


def summarize_node(state: AgentState) -> dict:
    style = detect_style(state["query"])
    answer = summarize(
        state["client"],
        state["chunks"],
        style=style,
        query=state["query"]
    )
    return {"answer": answer, "extra": None}


def quiz_node(state: AgentState) -> dict:
    questions = generate_quiz(
        state["client"],
        state["chunks"],
        num_questions=15
    )
    answer = (
        "Here's a quiz based on your material:"
        if questions
        else "I generated a quiz, but couldn't format it correctly. Please try again."
    )
    return {"answer": answer, "extra": questions}


def web_search_node(state: AgentState) -> dict:
    history_text = format_history(state["chat_history"])
    answer = answer_with_web_search(
        state["client"],
        state["query"],
        history_text,
    )
    return {"answer": answer, "extra": None}


def general_chat_node(state: AgentState) -> dict:
    history_text = format_history(state["chat_history"])

    system_prompt = (
        "You are a friendly study assistant. Answer the user's message "
        "helpfully and concisely."
    )

    user_prompt = (
        f"Recent conversation:\n{history_text}\n\nMessage: {state['query']}"
    )

    response = groq_call(
        state["client"],
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )

    return {"answer": response.choices[0].message.content, "extra": None}


def select_next_node(state: AgentState) -> str:
    return state["tool"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("doc_qa", doc_qa_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("general_chat", general_chat_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        select_next_node,
        {
            "doc_qa": "doc_qa",
            "summarize": "summarize",
            "quiz": "quiz",
            "web_search": "web_search",
            "general_chat": "general_chat",
        },
    )

    graph.add_edge("doc_qa", END)
    graph.add_edge("summarize", END)
    graph.add_edge("quiz", END)
    graph.add_edge("web_search", END)
    graph.add_edge("general_chat", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent_graph(
    client,
    query,
    embedder=None,
    index=None,
    chunks=None,
    chat_history=None,
):
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "chat_history": chat_history or [],
        "client": client,
        "embedder": embedder,
        "index": index,
        "chunks": chunks,
        "tool": None,
        "answer": None,
        "extra": None,
    }

    final_state = graph.invoke(initial_state)

    return final_state["tool"], final_state["answer"], final_state["extra"]