"""Simple conversation memory (Streamlit session state holds the Turn list)."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Turn:
    question: str
    answer: str
    sql: str = ""
    response: Optional[Any] = None   # full AnalystResponse, for UI replay


def build_history_messages(turns: list[Turn], max_turns: int = 3) -> list[dict]:
    """Recent turns as chat messages. The SQL is included so follow-ups like
    'what about August?' can reuse the same metric definition."""
    msgs: list[dict] = []
    for t in turns[-max_turns:]:
        msgs.append({"role": "user", "content": t.question})
        content = t.answer + (f"\n[SQL used previously: {t.sql}]" if t.sql else "")
        msgs.append({"role": "assistant", "content": content})
    return msgs
