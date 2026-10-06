"""Anti-hallucination layer for the result -> LLM interpretation step.

Two halves:
  1. ground_result()  : turns raw DB rows into an AUTHORITATIVE block (typed columns,
                        units, exact display strings, Python-computed derived values).
  2. verify_answer()  : deterministically checks that every number in the LLM's answer
                        exists in that block. If not -> repair once -> deterministic fallback.

Principle: the database is the source of truth; the LLM only writes the words around it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

RUPEE = "₹"

# ----------------------------------------------------------------- unit inference
_PERCENT = {"pct", "percent", "percentage", "rate", "growth", "share"}
_COUNT = {"count", "quantity", "qty", "orders", "customers", "units", "num", "number"}
_CURRENCY = {"revenue", "sales", "amount", "price", "cost", "value", "spend", "gmv", "aov", "income", "profit"}
_DATE = {"date", "month", "year", "day", "week", "quarter"}
_UNITS = {
    "currency": "INR (Indian Rupees, whole rupees)",
    "count": "count (number of items)",
    "percent": "percent (already on a 0-100 scale)",
    "number": "plain number",
    "id": "identifier (not a measure)",
    "date": "date / period label",
    "text": "",
}


@dataclass(frozen=True)
class ColumnMeta:
    name: str
    kind: str   # currency | count | percent | number | id | date | text
    unit: str


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def infer_column_meta(name: str, values: Sequence[Any]) -> ColumnMeta:
    tokens = set(re.split(r"[_\W]+", name.lower())) - {""}
    non_null = [v for v in values if v is not None]
    numeric = bool(non_null) and all(_is_num(v) for v in non_null)

    if not numeric:
        kind = "date" if tokens & _DATE else "text"
    elif "id" in tokens:
        kind = "id"
    elif tokens & _PERCENT:
        kind = "percent"
    elif tokens & _COUNT:
        kind = "count"
    elif tokens & _CURRENCY:
        kind = "currency"
    elif tokens & _DATE:
        kind = "date"
    else:
        kind = "number"
    return ColumnMeta(name, kind, _UNITS[kind])


# ----------------------------------------------------------------- formatting
def indian_group(n: int) -> str:
    """12345678 -> '1,23,45,678' (lakh/crore grouping, but written in full digits)."""
    s = str(abs(int(n)))
    if len(s) > 3:
        head, tail, parts = s[:-3], s[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + s


def format_decimal(value: float, max_decimals: int = 2, pad: bool = False) -> str:
    v = round(float(value), max_decimals)
    if v == int(v):
        return indian_group(int(v))
    int_part, frac = f"{abs(v):.{max_decimals}f}".split(".")
    if not pad:
        frac = frac.rstrip("0")
    return f"{'-' if v < 0 else ''}{indian_group(int(int_part))}.{frac}"


def format_value(value: Any, meta: ColumnMeta) -> str:
    if value is None:
        return "NULL"
    if not _is_num(value) or meta.kind in ("text", "date", "id"):
        return str(value)
    if meta.kind == "currency":
        body = format_decimal(abs(value), 2, pad=True)
        return f"{'-' if value < 0 else ''}{RUPEE}{body}"
    if meta.kind == "percent":
        return f"{format_decimal(value)}%"
    return format_decimal(value)


# ----------------------------------------------------------------- number extraction
_NUM_RE = re.compile(
    r"(?<![\w.])(?P<num>(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)"
    r"(?:\s?(?P<suffix>lakhs?|crores?|cr|mn|bn|million|billion|[lkm])\b)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumberToken:
    value: float
    raw: str
    suffix: Optional[str]


def extract_numbers(text: str) -> list[NumberToken]:
    return [NumberToken(float(m.group("num").replace(",", "")), m.group(0).strip(), m.group("suffix"))
            for m in _NUM_RE.finditer(text or "")]


# ----------------------------------------------------------------- grounding
@dataclass
class GroundedResult:
    text: str                      # the block sent to the LLM as the tool message
    allowed_numbers: set[float]    # every number the answer may legitimately contain
    shown_rows: int
    total_rows: int
    fallback_answer: str           # deterministic answer if the LLM cannot be verified
    metas: list[ColumnMeta]


HEADER = (
    "=== AUTHORITATIVE QUERY RESULT (from the database) ===\n"
    "Source: SQL executed on the business database. Every value below is exact, final and trusted.\n"
    "You MUST NOT alter, round, rescale, re-unit or re-calculate any value. Quote the 'display' form."
)


def _label(row: Sequence[Any], label_idx: Optional[int], j: int) -> str:
    return str(row[label_idx]) if label_idx is not None else f"Row {j + 1}"


def _signed(value: float, meta: ColumnMeta) -> str:
    text = format_value(value, meta)
    return text if value < 0 else f"+{text}"


def ground_result(columns: Sequence[str], rows: Sequence[Sequence[Any]], *,
                  db_truncated: bool = False, max_rows_for_llm: int = 25) -> GroundedResult:
    metas = [infer_column_meta(c, [r[i] for r in rows]) for i, c in enumerate(columns)]
    total, shown = len(rows), list(rows[:max_rows_for_llm])
    allowed: set[float] = set()

    def allow(v: float) -> None:
        allowed.add(float(v))
        allowed.add(abs(float(v)))

    def allow_text(s: str) -> None:
        for tok in extract_numbers(s):
            allow(tok.value)

    label_idx = next((i for i, m in enumerate(metas) if m.kind in ("text", "date")), None)

    # ---- columns
    lines = [HEADER, "", "COLUMNS"]
    for i, (c, m) in enumerate(zip(columns, metas), 1):
        lines.append(f"{i}. {c} | type: {m.kind}" + (f" | unit: {m.unit}" if m.unit else ""))

    # ---- rows
    if db_truncated:
        scope = f"showing {len(shown)} of at least {total} rows; the result was TRUNCATED by a row cap"
    elif total > len(shown):
        scope = f"showing first {len(shown)} of {total} rows; the remaining rows are not shown"
    else:
        scope = f"{total} of {total} rows; this is the COMPLETE result"
    lines += ["", f"ROWS ({scope})"]
    if not shown:
        lines.append("(no rows returned)")
    for n, row in enumerate(shown, 1):
        cells = []
        for c, m, v in zip(columns, metas, row):
            if v is None:
                cells.append(f"{c} = NULL")
            elif m.kind in ("text", "date"):
                allow_text(str(v))
                cells.append(f'{c} = "{v}"')
            elif m.kind == "id":
                allow(v)
                cells.append(f"{c} = {v} (identifier)")
            else:
                disp = format_value(v, m)
                allow(v)
                allow_text(disp)
                unit = {"currency": "INR", "count": "count", "percent": "percent"}.get(m.kind, "number")
                cells.append(f'{c} = {v} (display: "{disp}", unit: {unit})')
        lines.append(f"Row {n}: " + " | ".join(cells))

    # ---- derived values (computed in Python, never by the LLM)
    lines += ["", "DERIVED VALUES (calculated by Python; also authoritative)"]
    derived: list[str] = []
    if db_truncated:
        derived.append("- Totals/shares are NOT provided because the result was truncated. Do not state totals.")
    else:
        for i, (c, m) in enumerate(zip(columns, metas)):
            vals = [r[i] for r in rows]
            if m.kind not in ("currency", "count", "number") or not vals or not all(_is_num(v) for v in vals):
                continue
            n = len(vals)
            if n >= 2:
                tot = sum(vals)
                allow(tot)
                allow_text(format_value(tot, m))
                derived.append(f'- {c}: total across all {n} rows = "{format_value(tot, m)}"')
                if n <= 10 and m.kind in ("currency", "count") and tot > 0 and all(v >= 0 for v in vals):
                    shares = []
                    for j, v in enumerate(vals):
                        pct = round(v * 100.0 / tot, 2)
                        allow(pct)
                        shares.append(f'{_label(rows[j], label_idx, j)} {format_decimal(pct)}%')
                    derived.append(f"- {c}: share of total -> " + ", ".join(shares))
            if n == 2:
                a, b = vals
                diff = b - a
                allow(diff)
                allow_text(format_value(diff, m))
                l0, l1 = _label(rows[0], label_idx, 0), _label(rows[1], label_idx, 1)
                text = f'- {c}: change from "{l0}" to "{l1}" = "{_signed(diff, m)}"'
                if a != 0:
                    pct = round(diff * 100.0 / a, 2)
                    allow(pct)
                    allow_text(format_decimal(pct))
                    word = "increase" if diff > 0 else "decrease" if diff < 0 else "no change"
                    text += f' ({format_decimal(pct)}% {word})'
                derived.append(text)
    lines += derived or ["(none)"]

    allowed.update(float(k) for k in range(1, max(3, total) + 1))   # ordinals / "top N" phrasing

    return GroundedResult("\n".join(lines), allowed, len(shown), total,
                          _fallback_answer(columns, metas, shown, total, db_truncated), metas)


def _fallback_answer(columns, metas, shown, total, truncated) -> str:
    if total == 0:
        return "The query ran successfully but returned no rows, so there is no matching data."
    if total == 1 and len(columns) <= 3:
        parts = [f"{c}: {format_value(v, m)}" for c, m, v in zip(columns, metas, shown[0])]
        return "Result - " + "; ".join(parts) + "."
    head = f"The query returned {'at least ' if truncated else ''}{total} rows. First {min(5, len(shown))}:"
    body = ["- " + "; ".join(f"{c}: {format_value(v, m)}" for c, m, v in zip(columns, metas, r))
            for r in shown[:5]]
    return "\n".join([head, *body])


# ----------------------------------------------------------------- verification
@dataclass
class Verification:
    ok: bool
    violations: list[str]


def _in(v: float, allowed: set[float]) -> bool:
    return any(math.isclose(v, a, rel_tol=1e-9, abs_tol=1e-9) for a in allowed)


def verify_answer(answer: str, grounded: GroundedResult, question: str = "") -> Verification:
    """Every number in the answer must come from the authoritative block (or the question)."""
    allowed = set(grounded.allowed_numbers) | {t.value for t in extract_numbers(question)}
    bad: list[str] = []
    for tok in extract_numbers(answer):
        if tok.suffix:
            bad.append(f"'{tok.raw}' (abbreviated / converted unit)")
        elif 1900 <= tok.value <= 2100 and float(tok.value).is_integer():
            continue                                    # years are harmless
        elif not _in(tok.value, allowed):
            bad.append(f"'{tok.raw}' (not in the query result)")
    return Verification(not bad, bad)
