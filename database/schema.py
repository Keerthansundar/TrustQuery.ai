"""Live schema introspection (the source of truth for table/column names)."""
from typing import Optional

from database.db import get_connection


def get_table_names(path=None) -> list[str]:
    conn = get_connection(path=path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def get_schema_text(path=None) -> str:
    conn = get_connection(path=path)
    try:
        out = []
        for table in get_table_names(path):
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            fks = {r[3]: f"{r[2]}.{r[4]}" for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')}
            out.append(f"TABLE {table} ({count} rows)")
            for _, name, ctype, notnull, _, pk in conn.execute(f'PRAGMA table_info("{table}")'):
                extra = " PRIMARY KEY" if pk else ""
                extra += f" -> {fks[name]}" if name in fks else ""
                out.append(f"  {name} {ctype}{extra}")
            out.append("")
        return "\n".join(out).strip()
    finally:
        conn.close()


def get_data_window(path=None) -> Optional[tuple[str, str]]:
    conn = get_connection(path=path)
    try:
        lo, hi = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM orders").fetchone()
    finally:
        conn.close()
    return (lo, hi) if lo else None
