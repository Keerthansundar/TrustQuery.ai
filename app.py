"""TrustQuery AI - Streamlit UI.  Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

from agent.agent import DataAnalystAgent
from agent.memory import Turn, build_history_messages
from config import get_settings
from scripts.seed_database import build_database

st.set_page_config(page_title="TrustQuery AI", page_icon="🔎", layout="wide")
settings = get_settings()

EXAMPLES = [
    "What was our revenue in September?",
    "Which product had the highest revenue in September?",
    "Show me the top 5 customers by revenue.",
    "Why did revenue decrease in September compared with August?",
]
GROUNDING_BADGE = {
    "verified": ("success", "All numbers in this answer were verified against the query result."),
    "repaired": ("success", "Numbers verified (the answer was auto-corrected once)."),
    "fallback": ("warning", "The model's wording could not be verified, so this is a direct summary of the query result."),
}


@st.cache_resource
def get_agent() -> DataAnalystAgent:
    if not settings.db_path.exists():
        build_database(settings.db_path)
    return DataAnalystAgent()


def ollama_status(agent: DataAnalystAgent) -> tuple[bool, str]:
    try:
        names = [m.model for m in agent.client.list().models]
    except Exception:
        return False, f"Ollama is not reachable at {settings.ollama_host}. Start it with `ollama serve`."
    if settings.llm_model not in names:
        return False, f"Model not found. Run `ollama pull {settings.llm_model}`."
    return True, f"Ollama connected - {settings.llm_model}"


def ask(question: str) -> None:
    history = build_history_messages(st.session_state.turns, settings.history_turns)
    with st.spinner("Writing SQL, running it, and checking the answer..."):
        resp = get_agent().run(question, history)
    st.session_state.turns.append(Turn(question, resp.answer, resp.sql, resp))


def render_answer(turn: Turn) -> None:
    r = turn.response
    st.markdown(f"**You asked:** {turn.question}")
    if r.status in ("failed", "error"):
        st.error(r.answer)
        return
    if r.status == "clarification":
        st.info(r.answer)
        return
    st.subheader("Answer")
    st.markdown(r.answer)
    level, msg = GROUNDING_BADGE.get(r.grounding_status, ("info", ""))
    getattr(st, level)(msg, icon="✅" if level == "success" else "⚠️")
    if r.truncated:
        st.caption(f"Result capped at {settings.max_rows} rows.")
    with st.expander(f"Query result ({r.row_count} rows)"):
        st.dataframe(pd.DataFrame(r.rows, columns=r.columns), use_container_width=True, hide_index=True)
    if r.reasoning_summary:
        st.caption(f"How this was answered: {r.reasoning_summary}")
    st.markdown("**Sources**")
    for src in r.sources:
        st.markdown(f"- {src}")


def render_sidebar(turn: Turn | None, agent: DataAnalystAgent) -> None:
    with st.sidebar:
        st.header("SQL Reference")
        if turn and turn.response.executed:
            r = turn.response
            st.code(r.sql, language="sql")
            st.success("SQL validated", icon="✅")
            st.success("Query executed", icon="✅")
            if r.sql_attempts > 1:
                st.caption(f"Corrected after {r.sql_attempts - 1} retry.")
            st.subheader("Execution")
            t = r.timings_ms
            st.markdown(
                f"- SQL generation: **{t.get('llm_sql_ms', 0):.0f} ms**\n"
                f"- SQL execution: **{t.get('sql_exec_ms', 0):.0f} ms**\n"
                f"- Final response: **{t.get('llm_answer_ms', 0):.0f} ms**\n"
                f"- Total: **{t.get('total_ms', 0) / 1000:.2f} s**")
        elif turn:
            st.info("No SQL was executed for this question.")
        else:
            st.caption("The exact SQL that runs against the database will appear here.")
        st.divider()
        ok, msg = ollama_status(agent)
        (st.success if ok else st.error)(msg, icon="🟢" if ok else "🔴")
        st.caption("Phase 2: full docs in prompt. RAG arrives in Phase 3.")


# ----------------------------------------------------------------------------- page
agent = get_agent()
st.session_state.setdefault("turns", [])

st.title("TrustQuery AI")
st.caption("Ask business questions in plain English. Get trustworthy answers backed by executable SQL.")

with st.form("ask_form"):
    q = st.text_input("Ask a question about your business data",
                      placeholder="e.g. Why did revenue decrease in September?")
    submitted = st.form_submit_button("Ask", type="primary")

st.caption("Try one:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, use_container_width=True, key=f"ex_{ex}",
               on_click=lambda e=ex: st.session_state.update(pending=e))

question = st.session_state.pop("pending", None) or (q.strip() if submitted and q.strip() else None)
if question:
    ask(question)

turns: list[Turn] = st.session_state.turns
st.divider()
if turns:
    render_answer(turns[-1])
    if len(turns) > 1:
        with st.expander("Earlier questions"):
            for t in reversed(turns[:-1]):
                st.markdown(f"**{t.question}**  \n{t.answer}")
else:
    st.info("Ask a question above or pick an example to get started.")

render_sidebar(turns[-1] if turns else None, agent)
