import os
from dotenv import load_dotenv
load_dotenv()
import os
print("MONGODB_URI =", os.getenv("MONGODB_URI"))
import streamlit as st
from groq import Groq

from rag.pdf_loader import extract_text_from_multiple
from rag.chunker import chunk_documents, get_document_stats
from rag.embeddings import load_embedder
from rag.vector_store import build_vector_store
from utils.helper import confidence_color

from agents.router_agent import run_agent as run_agent_manual
from agents.langgraph_workflow import run_agent_graph
from agents.interview_agent import generate_interview_questions, evaluate_answer
from quiz.quiz_evaluator import evaluate_quiz, quiz_to_text

from database.db import init_db
from database.user_progress import (
    log_quiz_attempt,
    log_interview_attempt,
    log_document_processed,
)
from database.chat_history import save_message, load_history, clear_history
from database.mongo_client import mongo_available
from analytics.dashboard import render_dashboard

from auth.auth import signup, login, get_current_user



st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

/* Main page spacing */
.main {
    padding-top: 1rem;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
}

/* Inputs */
.stTextInput > div > div > input {
    border-radius: 10px;
}

/* Metrics */
div[data-testid="metric-container"] {
    border: 1px solid #e5e7eb;
    padding: 12px;
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    padding-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

init_db()

defaults = {
    "auth_token": None,
    "username": None,
    "embedder": None,
    "index": None,
    "chunks": None,
    "doc_stats": None,
    "messages": [],
    "quiz_results": {},
    "interview_questions": [],
    "interview_results": {},
    "use_langgraph": False,
    
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def render_auth_screen():
    st.title("📚 AI Study & Research Assistant")
    st.caption("Please log in or create an account to continue")

    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            success, result = login(username, password)
            if success:
                st.session_state.auth_token = result
                st.session_state.username = username
                st.rerun()
            else:
                st.error(result)

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create account")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = signup(new_username, new_password)
                if success:
                    st.success(message + " You can now log in.")
                else:
                    st.error(message)

    if not mongo_available():
        st.info(
            "ℹ️ MongoDB is not configured (set the MONGODB_URI environment "
            "variable to enable it). Running with local file storage for "
            "now — fine for development/demo."
        )


if not st.session_state.auth_token:
    render_auth_screen()
    st.stop()

username = st.session_state.username

if not st.session_state.messages:
    past_messages = load_history(username)
    st.session_state.messages = [
        {
            "role": m["role"],
            "content": m["content"],
            "tool": m.get("tool"),
            "extra": m.get("extra"),
        }
        for m in past_messages
    ]

if st.session_state.embedder is None:
    with st.spinner("Loading embedding model... (first time only)"):
        st.session_state.embedder = load_embedder()


with st.sidebar:
    st.success(f"Logged in as **{username}**")

    if st.button("Log out", use_container_width=True):
        st.session_state.auth_token = None
        st.session_state.username = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    with st.expander("⚙️ Settings", expanded=True):

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
        )

        st.session_state.use_langgraph = st.toggle(
            "Use LangGraph workflow",
            value=st.session_state.use_langgraph,
        )
    

    st.divider()
    st.header("📄 Upload Study Material")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.columns(2)
    process_button = col_a.button("Process", use_container_width=True)
    clear_button = col_b.button("Clear chat", use_container_width=True)


if clear_button:
    st.session_state.messages = []
    st.session_state.quiz_results = {}
    clear_history(username)
    st.rerun()


if process_button:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Processing documents..."):
            documents = extract_text_from_multiple(uploaded_files)
            chunks = chunk_documents(documents)
            index = build_vector_store(chunks, st.session_state.embedder)
            stats = get_document_stats(documents, chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.doc_stats = stats
            st.session_state.interview_questions = []
            st.session_state.interview_results = {}

            for doc in documents:
                doc_chunks = [c for c in chunks if c["source"] == doc["filename"]]
                log_document_processed(
                    doc["filename"],
                    len(doc["text"].split()),
                    len(doc_chunks),
                )

        st.sidebar.success(
            f"Processed {stats['document_count']} file(s) into {stats['chunk_count']} chunks"
        )


if st.session_state.doc_stats:
    st.sidebar.divider()
    st.sidebar.subheader("📊 Document analytics")

    stats = st.session_state.doc_stats
    c1, c2 = st.sidebar.columns(2)

    c1.metric("Documents", stats["document_count"])
    c2.metric("Chunks", stats["chunk_count"])
    c1.metric("Words", f"{stats['word_count']:,}")
    c2.metric("Read time", f"{stats['reading_time_min']} min")

    with st.sidebar.expander("Files processed"):
        for fname in stats["filenames"]:
            st.caption(f"📄 {fname}")


st.title("📚 AI Study Assistant Pro")

st.markdown("""
### 🚀 Multi-Agent Learning Platform

✅ Multi-PDF RAG Chat

✅ LangGraph Workflow

✅ Mock Interview Generator

✅ Progress Analytics

✅ Chat History & Authentication
""")

st.caption(
    "AI-powered study assistant built with RAG, LangGraph, Groq and MongoDB"
)
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📄 Documents",
    st.session_state.doc_stats["document_count"]
    if st.session_state.doc_stats else 0
)

col2.metric(
    "🧩 Chunks",
    st.session_state.doc_stats["chunk_count"]
    if st.session_state.doc_stats else 0
)

col3.metric(
    "💬 Messages",
    len(st.session_state.messages)
)

col4.metric(
    "🎤 Interviews",
    len(st.session_state.interview_results)
)

tab_chat, tab_interview, tab_progress = st.tabs(
    ["💬 Chat", "🎤 Mock Interview", "📈 Progress"]
)


with tab_chat:

    if not st.session_state.chunks:
        st.info("""
👋 Welcome to AI Study Assistant Pro

Upload one or more PDFs and click Process.

Features:
• Multi-PDF Question Answering
• RAG-based Search
• LangGraph Workflow
• Mock Interviews
• Progress Analytics
""")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):

            if msg["role"] == "assistant" and msg.get("tool"):
                st.caption(f"🔧 Tool used: `{msg['tool']}`")

            if msg["role"] == "assistant":
                with st.container(border=True):
                    st.write(msg["content"])

            else:
                st.write(msg["content"])
    

    final_query = st.chat_input("Ask something...")

    if final_query:
        if not api_key:
            st.error("Enter Groq API key")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": final_query})
        save_message(username, "user", final_query)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        client = Groq(api_key=api_key)
        agent_runner = run_agent_graph if st.session_state.use_langgraph else run_agent_manual

        with st.spinner("Thinking..."):
            tool_used, answer, extra = agent_runner(
                client,
                final_query,
                embedder=st.session_state.embedder,
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                chat_history=history,
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "tool": tool_used, "extra": extra}
        )

        save_message(username, "assistant", answer, tool=tool_used, extra=extra)
        st.rerun()


with tab_interview:
    st.markdown("Mock interview system")

    if st.session_state.chunks:

        num_questions = st.slider(
            "Number of Questions",
            min_value=5,
            max_value=30,
            value=30
        )

        if st.button("Generate New Questions"):
            st.session_state.interview_questions = []
            st.session_state.interview_results = {}

            client = Groq(api_key=api_key)

            questions = generate_interview_questions(
                client,
                st.session_state.chunks,
                n=num_questions
            )

            st.session_state.interview_questions = questions

        for i, q in enumerate(st.session_state.interview_questions):
            st.markdown(f"**Q{i+1}. {q}**")

            answer = st.text_area(
                "Your answer",
                key=f"ans_{i}"
            )

            if st.button(
                "Submit",
                key=f"submit_{i}"
            ):
                client = Groq(api_key=api_key)

                result = evaluate_answer(
                    client,
                    q,
                    answer
                )

                st.session_state.interview_results[i] = result

                log_interview_attempt(
                    q,
                    answer,
                    result["rating"],
                    result["feedback"]
                )

            if i in st.session_state.interview_results:
                st.info(
                    st.session_state.interview_results[i]["feedback"]
                )

    else:
        st.info("Upload docs first")


with tab_progress:
    render_dashboard()