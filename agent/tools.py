"""Controlled tools.

The LLM never touches the database directly:

LLM
    -> tool call / SQL text
    -> Python validation
    -> read-only SQLite
    -> ToolResult
    -> LLM
"""

import json
import re
import sqlite3
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from database.db import QueryTimeoutError, run_query
from database.sql_validator import validate_sql
from models.response import ToolResult


# ---------------------------------------------------------------------------
# Ollama tool definition
# ---------------------------------------------------------------------------

EXECUTE_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": (
            "Execute ONE read-only SQLite SELECT query "
            "(WITH/CTE allowed) against the business database "
            "and return the result. Use this for every database fact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single SQLite SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    },
}


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def execute_sql(sql: str) -> ToolResult:
    """Validate and execute one read-only SQL query."""

    check = validate_sql(sql)

    if not check.is_valid:
        return ToolResult(
            success=False,
            sql=(sql or "").strip(),
            error=check.error,
            error_type="validation",
        )

    try:
        out = run_query(check.sql)

    except QueryTimeoutError as exc:
        return ToolResult(
            success=False,
            sql=check.sql,
            error=str(exc),
            error_type="timeout",
        )

    except sqlite3.Error as exc:
        return ToolResult(
            success=False,
            sql=check.sql,
            error=f"SQLite error: {exc}",
            error_type="execution",
        )

    return ToolResult(
        success=True,
        sql=check.sql,
        columns=out.columns,
        rows=out.rows,
        row_count=len(out.rows),
        truncated=out.truncated,
        elapsed_ms=out.elapsed_ms,
    )


# ---------------------------------------------------------------------------
# SQL extraction patterns
# ---------------------------------------------------------------------------

# ```sql
# SELECT ...
# ```
_FENCE = re.compile(
    r"```(?:sql)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

# <tool_call> ... </tool_call>
_TOOL_TAGS = re.compile(
    r"</?tool_call>",
    re.IGNORECASE,
)

# Supports:
#
# SQL: SELECT ...
# [SQL: SELECT ...]
# execute_sql: SELECT ...
# query: SELECT ...
#
# anywhere in the text.
_LABELLED = re.compile(
    r"\b(?:execute_sql|sql|query)\s*:\s*\(?\s*((?:select|with)\b[^\]]*)",
    re.IGNORECASE | re.DOTALL,
)

# SQLGlot AST types that represent SELECT-style queries.
_QUERY_ROOTS = (
    exp.Select,
    exp.Union,
    exp.Except,
    exp.Intersect,
)

# Money, grouped numbers, long numbers or percentages:
# these can indicate that an LLM has invented database results.
_MADE_UP = re.compile(
    r"₹|\d{1,3}(?:,\d{2,3})+|\b\d{5,}\b|\d+(?:\.\d+)?\s?%",
)


def looks_like_made_up_results(text: str) -> bool:
    """Return True if a text-only reply contains result-like numbers.

    A genuine clarifying question generally does not need database-result
    numbers. This helper can therefore be used by the agent layer to prevent
    hallucinated numerical answers from being shown to the user.

    Examples detected:
        ₹50,000
        12,500
        123456
        15%
        42.5%
    """

    return bool(_MADE_UP.search(text or ""))


# ---------------------------------------------------------------------------
# SQL normalization
# ---------------------------------------------------------------------------

def _normalize_sql(sql: str) -> str:
    """Normalize common LLM-generated SQL variations."""

    sql = sql.replace("≥", ">=")
    sql = sql.replace("≤", "<=")
    sql = sql.replace("≠", "!=")

    # Remove non-breaking spaces.
    sql = sql.replace("\u00a0", " ")

    return sql.strip()


# ---------------------------------------------------------------------------
# SQL extraction
# ---------------------------------------------------------------------------

def extract_sql_from_text(text: str) -> Optional[str]:
    """Extract SQL when a local LLM outputs SQL as plain text.

    Supported formats include:

        SQL: SELECT ...

        [SQL: SELECT ...]

        execute_sql: SELECT ...

        query: SELECT ...

        ```sql
        SELECT ...
        ```

        {"arguments": {"sql": "SELECT ..."}}

        {"parameters": {"sql": "SELECT ..."}}

        {"sql": "SELECT ..."}

    The extracted SQL is returned to agent.py.

    It MUST then be passed through execute_sql(), which performs the
    final validation before touching the database.

    Normal prose such as a clarifying question returns None and is
    never treated as SQL.
    """

    # ------------------------------------------------------------------
    # Basic cleanup
    # ------------------------------------------------------------------

    t = _TOOL_TAGS.sub("", (text or "")).strip()

    if not t:
        return None

    # ------------------------------------------------------------------
    # JSON tool-call style output
    # ------------------------------------------------------------------

    if t[0] in "{[" and not re.match(
        r"\[\s*(?:execute_sql|sql|query)\s*:",
        t,
        re.IGNORECASE,
    ):
        try:
            obj = json.loads(t)

            if isinstance(obj, list) and obj:
                obj = obj[0]

            if isinstance(obj, dict):

                # Handle:
                # {"arguments": {"sql": "..."}}
                # {"parameters": {"sql": "..."}}
                args = (
                    obj.get("arguments")
                    or obj.get("parameters")
                    or {}
                )

                if isinstance(args, dict) and "sql" in args:
                    return _normalize_sql(str(args["sql"]))

                # Also handle:
                # {"sql": "..."}
                if "sql" in obj:
                    return _normalize_sql(str(obj["sql"]))

        except (
            ValueError,
            AttributeError,
            TypeError,
            KeyError,
        ):
            return None

    # ------------------------------------------------------------------
    # Detect labelled SQL or fenced SQL
    #
    # Examples:
    #
    # SQL: SELECT ...
    #
    # execute_sql: SELECT ...
    #
    # query: SELECT ...
    #
    # ```sql
    # SELECT ...
    # ```
    # ------------------------------------------------------------------

    labelled = _LABELLED.search(t)
    fence = _FENCE.search(t)

    if labelled:
        t = labelled.group(1)

    elif fence:
        t = fence.group(1)

    # ------------------------------------------------------------------
    # Remove common SQL prefixes
    #
    # This handles cases that weren't caught by _LABELLED.
    # ------------------------------------------------------------------

    t = re.sub(
        r"^\s*(?:execute_sql|sql|query)\s*[:(]\s*",
        "",
        t.strip(),
        flags=re.IGNORECASE,
    )

    t = t.strip()

    # ------------------------------------------------------------------
    # Remove explanation after a blank line.
    #
    # Example:
    #
    # SELECT ...
    #
    # This query calculates September revenue...
    #
    # Keep only SQL.
    # ------------------------------------------------------------------

    t = t.split("\n\n")[0].strip()

    if not t:
        return None

    # ------------------------------------------------------------------
    # Normalize Unicode SQL operators.
    #
    # Qwen may produce:
    #
    # order_date ≥ '2026-09-01'
    #
    # Convert to:
    #
    # order_date >= '2026-09-01'
    # ------------------------------------------------------------------

    t = _normalize_sql(t)

    # ------------------------------------------------------------------
    # SQL must begin with SELECT or WITH.
    # ------------------------------------------------------------------

    if not re.match(r"(?is)^(select|with)\b", t):
        return None

    # ------------------------------------------------------------------
    # Parse using SQLGlot.
    #
    # This checks whether the text resembles valid SQLite SQL.
    #
    # Actual security validation happens inside validate_sql().
    # ------------------------------------------------------------------

    try:
        trees = [
            tree
            for tree in sqlglot.parse(t, read="sqlite")
            if tree is not None
        ]

    except SqlglotError:
        return None

    # ------------------------------------------------------------------
    # No valid SQL AST.
    # ------------------------------------------------------------------

    if not trees:
        return None

    # ------------------------------------------------------------------
    # Only allow SELECT-style statements here.
    #
    # Multi-statement text is intentionally left to validate_sql(),
    # which should reject it explicitly.
    # ------------------------------------------------------------------

    if isinstance(trees[0], _QUERY_ROOTS):
        return t

    return None
