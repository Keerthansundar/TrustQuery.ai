"""SQLite access: always read-only, always with a timeout and a row cap."""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import get_settings


class QueryTimeoutError(RuntimeError):
    pass


@dataclass
class QueryOutcome:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    elapsed_ms: float


def get_connection(*, read_only: bool = True, path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = Path(path or get_settings().db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run scripts/seed_database.py")
    if read_only:
        # Layer 1 of defence: the OS-level connection itself cannot write.
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")  # Layer 2
        return conn
    return sqlite3.connect(db_path)


def run_query(sql: str, *, max_rows: Optional[int] = None, timeout_s: Optional[float] = None,
              path: Optional[Path] = None) -> QueryOutcome:
    settings = get_settings()
    max_rows = max_rows or settings.max_rows
    timeout_s = timeout_s or settings.query_timeout_s

    conn = get_connection(read_only=True, path=path)
    deadline = time.monotonic() + timeout_s
    # SQLite calls this every N VM steps; returning non-zero aborts the query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)

    start = time.perf_counter()
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        fetched = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as exc:
        if time.monotonic() > deadline:
            raise QueryTimeoutError(f"Query exceeded the {timeout_s:g}s time limit") from exc
        raise
    finally:
        conn.close()

    truncated = len(fetched) > max_rows
    rows = [list(r) for r in fetched[:max_rows]]
    return QueryOutcome(columns, rows, truncated, (time.perf_counter() - start) * 1000)
