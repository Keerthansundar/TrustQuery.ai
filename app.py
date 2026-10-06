import os
from pathlib import Path

import streamlit as st

from agent.agent import DataAnalystAgent
from agent.memory import Turn, build_followup_context
from config import get_settings


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="TrustQuery AI",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

settings = get_settings()


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

DB_PATH = Path(settings.db_path)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "turns" not in st.session_state:
    st.session_state.turns = []


# ---------------------------------------------------------
# AGENT
# ---------------------------------------------------------

@st.cache_resource
def get_agent() -> DataAnalystAgent:
    return DataAnalystAgent()


# ---------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------

def ensure_database() -> None:
    """Create the database if it does not already exist."""

    if DB_PATH.exists():
        return

    try:
        from scripts.seed_database import build_database

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_database(str(DB_PATH))

    except Exception as exc:
        st.error(f"Could not initialize database: {exc}")


# ---------------------------------------------------------
# OLLAMA STATUS
# ---------------------------------------------------------

def check_ollama_status() -> tuple[bool, str]:
    """Check whether Ollama is reachable."""

    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)

        response = client.list()

        models = []

        if hasattr(response, "models"):
            models = response.models

        model_names = []

        for model in models:
            if hasattr(model, "model"):
                model_names.append(model.model)
            elif isinstance(model, dict):
                model_names.append(model.get("name", ""))

        if settings.llm_model in model_names:
            return True, f"Ollama connected — {settings.llm_model}"

        return True, (
            f"Ollama connected, but model '{settings.llm_model}' "
            "was not found."
        )

    except Exception as exc:
        return False, f"Ollama unavailable: {exc}"


# ---------------------------------------------------------
# ASK AGENT
# ---------------------------------------------------------

def ask(question: str) -> None:
    """Send a question to TrustQuery AI."""

    question = question.strip()

    if not question:
        return

    context = build_followup_context(
        st.session_state.turns,
        settings.history_turns,
    )

    with st.spinner(
        "Writing SQL, running it, and checking the answer..."
    ):
        try:
            resp = get_agent().run(
                question,
                context,
            )

            # Store conversation turn
            st.session_state.turns.append(
                Turn(
                    question=question,
                    answer=resp.answer,
                    sql=resp.sql or "",
                    response=resp,
                )
            )

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


# ---------------------------------------------------------
# RENDER ANSWER
# ---------------------------------------------------------

def render_answer(resp) -> None:
    """Render an AnalystResponse."""

    if resp is None:
        return

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    if resp.status == "success":
        st.success("Query executed successfully")

    elif resp.status == "failed":
        st.warning("The query could not be executed.")

    elif resp.status == "error":
        st.error("An error occurred while processing the question.")

    # -----------------------------------------------------
    # Grounding status
    # -----------------------------------------------------

    grounding_status = getattr(
        resp,
        "grounding_status",
        None,
    )

    if grounding_status:
        if grounding_status == "verified":
            st.caption("🟢 Answer verified against database results")

        elif grounding_status == "repaired":
            st.caption("🟡 Answer repaired and verified")

        elif grounding_status == "fallback":
            st.caption("🟠 Deterministic fallback used")

        else:
            st.caption(
                f"Grounding status: {grounding_status}"
            )

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    st.markdown("### Answer")

    if resp.answer:
        st.write(resp.answer)
    else:
        st.write("No answer was generated.")

    # -----------------------------------------------------
    # Query Result
    # -----------------------------------------------------

    columns = getattr(resp, "columns", None)
    rows = getattr(resp, "rows", None)

    if columns and rows is not None:

        st.markdown("### Query Result")

        try:
            import pandas as pd

            dataframe = pd.DataFrame(
                rows,
                columns=columns,
            )

            st.dataframe(
                dataframe,
                use_container_width=True,
            )

        except Exception:
            st.write(rows)

    # -----------------------------------------------------
    # Reasoning Summary
    # -----------------------------------------------------

    reasoning_summary = getattr(
        resp,
        "reasoning_summary",
        None,
    )

    if reasoning_summary:
        with st.expander("Reasoning Summary"):
            st.write(reasoning_summary)

    # -----------------------------------------------------
    # Data Used
    # -----------------------------------------------------

    data_used = getattr(
        resp,
        "data_used",
        None,
    )

    if data_used:
        with st.expander("Data Used"):
            for item in data_used:
                st.write(f"- {item}")

    # -----------------------------------------------------
    # Errors
    # -----------------------------------------------------

    error = getattr(
        resp,
        "error",
        None,
    )

    if error:
        with st.expander("Details"):
            st.error(error)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def render_sidebar() -> None:

    with st.sidebar:

        st.title("🔎 TrustQuery AI")

        st.caption(
            "Natural-language business analytics powered by "
            "Qwen2.5 + SQLite"
        )

        st.divider()

        # -------------------------------------------------
        # Ollama
        # -------------------------------------------------

        st.subheader("Ollama")

        ollama_ok, ollama_message = check_ollama_status()

        if ollama_ok:
            st.success(ollama_message)
        else:
            st.error(ollama_message)

        st.divider()

        # -------------------------------------------------
        # Latest SQL
        # -------------------------------------------------

        st.subheader("Executed SQL")

        if st.session_state.turns:

            latest_turn = st.session_state.turns[-1]

            if latest_turn.sql:

                st.code(
                    latest_turn.sql,
                    language="sql",
                )

            else:
                st.caption(
                    "No SQL was executed for this question."
                )

            # -------------------------------------------------
            # Execution information
            # -------------------------------------------------

            resp = latest_turn.response

            if resp is not None:

                st.markdown("### Query Status")

                executed = getattr(
                    resp,
                    "executed",
                    False,
                )

                if executed:
                    st.success("Executed")

                else:
                    st.warning("Not executed")

                # SQL attempts
                sql_attempts = getattr(
                    resp,
                    "sql_attempts",
                    None,
                )

                if sql_attempts is not None:
                    st.caption(
                        f"SQL attempts: {sql_attempts}"
                    )

                # Answer repairs
                answer_repairs = getattr(
                    resp,
                    "answer_repairs",
                    None,
                )

                if answer_repairs is not None:
                    st.caption(
                        f"Answer repairs: {answer_repairs}"
                    )

                # Timing
                timings = getattr(
                    resp,
                    "timings_ms",
                    None,
                )

                if timings:

                    st.markdown("### Timing")

                    for key, value in timings.items():

                        label = key.replace(
                            "_",
                            " ",
                        ).title()

                        st.caption(
                            f"{label}: {value} ms"
                        )

        else:

            st.caption(
                "Ask a question to see the executed SQL."
            )

        st.divider()

        # -------------------------------------------------
        # Clear history
        # -------------------------------------------------

        if st.button(
            "Clear Conversation",
            use_container_width=True,
        ):
            st.session_state.turns = []
            st.rerun()


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------

ensure_database()

render_sidebar()


st.title("🔎 TrustQuery AI")

st.markdown(
    """
Ask business questions in plain English and TrustQuery AI will:

1. Generate a read-only SQL query
2. Validate the SQL using SQLGlot
3. Execute it against SQLite
4. Ground the answer using the database result
5. Verify the generated answer
6. Repair or fall back if the answer is inconsistent
"""
)


# ---------------------------------------------------------
# EXAMPLE QUESTIONS
# ---------------------------------------------------------

st.markdown("### Try an example")

examples = [
    "What was our revenue in September?",
    "Which product had the highest revenue in September?",
    "Show the top 5 customers by revenue.",
    "Why did revenue decrease in September compared with August?",
]

cols = st.columns(2)

for index, example in enumerate(examples):

    with cols[index % 2]:

        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):
            ask(example)
            st.rerun()


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a business question..."
)


if question:

    # Display user question
    with st.chat_message("user"):
        st.write(question)

    # Execute agent
    ask(question)

    # Refresh UI
    st.rerun()


# ---------------------------------------------------------
# CONVERSATION HISTORY
# ---------------------------------------------------------

for turn in st.session_state.turns:

    with st.chat_message("user"):
        st.write(turn.question)

    with st.chat_message("assistant"):
        render_answer(turn.response)