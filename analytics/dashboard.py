import streamlit as st
import pandas as pd

from database.user_progress import get_progress_summary, get_interview_history
from quiz.quiz_analytics import get_quiz_trend, get_best_and_worst


def render_dashboard():
    st.subheader("📈 Your Learning Progress")

    summary = get_progress_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents processed", summary["documents_processed"])
    col2.metric("Quizzes taken", summary["quiz_attempts"])
    col3.metric("Avg. quiz score", f"{summary['avg_quiz_percentage']}%")
    col4.metric("Avg. interview rating", f"{summary['avg_interview_rating']}/5")

    st.divider()

    trend = get_quiz_trend()
    if trend:
        st.markdown("**Quiz score trend (over time)**")
        df = pd.DataFrame(trend)[["attempt_number", "percentage"]].set_index("attempt_number")
        st.line_chart(df)

        best, worst = get_best_and_worst()
        c1, c2 = st.columns(2)
        c1.success(f"🏆 Best attempt: {best['percentage']}% on {best['timestamp']}")
        c2.warning(f"📉 Lowest attempt: {worst['percentage']}% on {worst['timestamp']}")
    else:
        st.info("No quizzes taken yet. Try asking the assistant to 'generate a quiz' in the Chat tab!")

    st.divider()

    st.markdown("**Recent mock interview feedback**")
    interviews = get_interview_history(limit=5)
    if interviews:
        for item in interviews:
            with st.expander(f"Q: {item['question'][:80]}... — Rating: {item['rating']}/5"):
                st.markdown(f"**Your answer:** {item['answer']}")
                st.markdown(f"**Feedback:** {item['feedback']}")
    else:
        st.info("No mock interview attempts yet. Try the 'Mock Interview' tab!")