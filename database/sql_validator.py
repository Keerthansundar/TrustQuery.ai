"""AST-based SQL safety validation (sqlglot). Stricter and safer than keyword regexes.

Rules: exactly one statement; root must be SELECT / UNION-style query (CTEs allowed);
no DML/DDL/PRAGMA/ATTACH anywhere in the tree; only known tables; no dangerous functions.
"""
from dataclasses import dataclass, field
from typing import Iterable, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)
FORBIDDEN_NODES = {
    "Insert", "Update", "Delete", "Drop", "Alter", "AlterTable", "Create", "TruncateTable",
    "Command", "Pragma", "Attach", "Detach", "Merge", "Transaction", "Commit", "Rollback",
    "Set", "Use", "Copy", "Into",
}
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass
class ValidationResult:
    is_valid: bool
    sql: str = ""                       # normalized SQL that is safe to execute
    error: Optional[str] = None
    tables: list[str] = field(default_factory=list)


def _fail(sql: str, msg: str) -> ValidationResult:
    return ValidationResult(False, sql, msg)


def validate_sql(sql: str, allowed_tables: Optional[Iterable[str]] = None) -> ValidationResult:
    cleaned = (sql or "").strip().rstrip(";").strip()
    if not cleaned:
        return _fail(sql, "Empty SQL.")

    try:
        statements = [s for s in sqlglot.parse(cleaned, read="sqlite") if s is not None]
    except SqlglotError as exc:
        return _fail(cleaned, f"SQL could not be parsed: {str(exc).splitlines()[0]}")

    if len(statements) != 1:
        return _fail(cleaned, "Only a single SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, ALLOWED_ROOTS):
        return _fail(cleaned, f"Only read-only SELECT queries are allowed (got {type(tree).__name__}).")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables: list[str] = []
    for node in tree.find_all(exp.Expression):
        kind = type(node).__name__
        if kind in FORBIDDEN_NODES:
            return _fail(cleaned, f"Forbidden operation in query: {kind.upper()}.")
        if isinstance(node, exp.Anonymous) and node.name.lower() in FORBIDDEN_FUNCTIONS:
            return _fail(cleaned, f"Forbidden function: {node.name}.")
        if isinstance(node, exp.Table):
            if node.db and node.db.lower() != "main":
                return _fail(cleaned, f"Schema-qualified table '{node.db}.{node.name}' is not allowed.")
            name = node.name.lower()
            if name and name not in cte_names:
                tables.append(name)

    if allowed_tables is None:
        try:
            from database.schema import get_table_names
            allowed_tables = get_table_names()
        except Exception:  # DB unavailable -> skip table allow-list (read-only conn still protects)
            allowed_tables = None
    if allowed_tables is not None:
        allowed = {t.lower() for t in allowed_tables}
        unknown = sorted({t for t in tables if t not in allowed})
        if unknown:
            return _fail(cleaned, f"Unknown table(s): {', '.join(unknown)}. Available: {', '.join(sorted(allowed))}.")

    return ValidationResult(True, cleaned, None, sorted(set(tables)))
