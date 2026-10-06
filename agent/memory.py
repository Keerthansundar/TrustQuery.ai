"""Simple conversation memory.

Streamlit session state holds the Turn list.

Earlier turns are NOT replayed as assistant messages because small models
may imitate them and produce fake answers instead of calling the tool.

Instead, previous questions and the SQL that actually ran are passed as a
short context note inside the new user message.

Previous answers/numbers are deliberately left out.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Turn:
    question: str
    answer: str
    sql: str = ""
    response: Optional[Any] = None  # Full AnalystResponse for UI replay


def build_followup_context(
    turns: list[Turn],
    max_turns: int = 3,
) -> str:
    """Build short context from recent turns for follow-up questions."""

    recent = [turn for turn in turns[-max_turns:] if turn.sql]

    if not recent:
        return ""

    lines = [
        "EARLIER IN THIS CONVERSATION "
        "(only to resolve follow-ups such as 'what about August?'. "
        "Do not reuse any numbers; always run a NEW query with execute_sql):"
    ]

    for turn in recent:
        lines.append(
            f"- Earlier question: {turn.question}\n"
            f"  SQL that was run: {turn.sql}"
        )

    return "\n".join(lines)