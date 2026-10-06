"""Deterministic planner + pre-flight checks. No LLM call, so zero extra latency.

preflight()       -> refuse writes / ask for clarification BEFORE any model runs
is_out_of_scope() -> abstain only when retrieval AND vocabulary both say "unrelated"
make_plan()       -> intent, resolved date ranges, hints injected into the prompt
"""
from __future__ import annotations

import calendar
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

from config import get_settings
from database.db import get_connection

# ----------------------------------------------------------------------------- messages
REFUSAL = ("TrustQuery is read-only: I can analyse your data but I can't delete, change or insert "
           "records. Ask me a question about your orders, customers or products instead.")
OUT_OF_SCOPE = ("I can only answer questions about this company's orders, customers and products, and I "
                "couldn't find anything in the data or business rules related to that question. Try asking "
                "about revenue, orders, customers, products or categories.")


@dataclass
class Preflight:
    status: str      # "clarification" | "refused"
    message: str


# ----------------------------------------------------------------------------- periods
@dataclass(frozen=True)
class Period:
    label: str
    start: date
    end: date        # exclusive

    def sql(self) -> str:
        return f"order_date >= '{self.start}' AND order_date < '{self.end}'"


_MONTHS = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
           "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sept": 9,
           "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
_MONTH_RE = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b")
_MAY_RE = re.compile(r"\b(?:in|of|for|during|from|to|and|vs|versus|than|with|since)\s+may\b|\bmay\s+20\d{2}\b")
_ORDINAL_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _month_end(d: date) -> bool:
    return (d + timedelta(days=1)).day == 1


def month_period(y: int, m: int) -> Period:
    return Period(f"{calendar.month_name[m]} {y}", date(y, m, 1), date(y + (m == 12), m % 12 + 1, 1))


def quarter_period(y: int, q: int) -> Period:
    last = 3 * q
    return Period(f"Q{q} {y}", date(y, last - 2, 1), date(y + (last == 12), last % 12 + 1, 1))


def resolve_periods(question: str, data_max: date) -> list[Period]:
    """Turn 'September', 'last month', 'Q3', 'this year' ... into exact date ranges (relative terms are
    resolved against the latest data date, not the system clock)."""
    q = question.lower()
    found: list[tuple[int, Period]] = []
    ym = re.search(r"\b(20\d{2})\b", q)
    year = int(ym.group(1)) if ym else data_max.year

    for m in re.finditer(r"\blast month\b", q):
        ref = data_max if _month_end(data_max) else data_max.replace(day=1) - timedelta(days=1)
        found.append((m.start(), month_period(ref.year, ref.month)))
    for m in re.finditer(r"\bthis month\b", q):
        found.append((m.start(), month_period(data_max.year, data_max.month)))
    for m in re.finditer(r"\blast quarter\b", q):
        cur = (data_max.month - 1) // 3 + 1
        if _month_end(data_max) and data_max.month % 3 == 0:
            yq = (data_max.year, cur)
        else:
            yq = (data_max.year - 1, 4) if cur == 1 else (data_max.year, cur - 1)
        found.append((m.start(), quarter_period(*yq)))
    for m in re.finditer(r"\bq([1-4])\b", q):
        found.append((m.start(), quarter_period(year, int(m.group(1)))))
    for m in re.finditer(r"\b(first|second|third|fourth) quarter\b", q):
        found.append((m.start(), quarter_period(year, _ORDINAL_Q[m.group(1)])))
    for m in _MONTH_RE.finditer(q):
        found.append((m.start(), month_period(year, _MONTHS[m.group(1)])))
    for m in _MAY_RE.finditer(q):
        found.append((m.end() - 3, month_period(year, 5)))
    for m in re.finditer(r"\b(this year|year to date|year-to-date|ytd|so far)\b", q):
        found.append((m.start(), Period(f"{data_max.year} so far", date(data_max.year, 1, 1),
                                        data_max + timedelta(days=1))))
    for m in re.finditer(r"\blast year\b", q):
        found.append((m.start(), Period(str(data_max.year - 1), date(data_max.year - 1, 1, 1),
                                        date(data_max.year, 1, 1))))
    if not found and ym:
        found.append((ym.start(), Period(str(year), date(year, 1, 1), date(year + 1, 1, 1))))

    out: list[Period] = []
    for _, p in sorted(found, key=lambda x: x[0]):
        if p not in out:
            out.append(p)
    return out


# ----------------------------------------------------------------------------- intent / plan
_WHY = re.compile(r"\b(why|reasons?|causes?|caused|explain|drivers?|what happened)\b")
_CHANGE = re.compile(r"\b(decreas\w*|drop\w*|declin\w*|fell|fall\w*|increas\w*|rose|rise|grew|growth|dip\w*|spike\w*)\b")
_COMPARE = re.compile(r"\b(compare|compared|comparison|versus|vs\.?|difference|differ)\b")
_RANK = re.compile(r"\b(top|best|highest|lowest|worst|most|least|rank\w*|leading|biggest|largest|smallest)\b")
_TREND = re.compile(r"\b(monthly|month by month|per month|each month|by month|trend|over time|weekly|daily|quarterly)\b")


@dataclass
class Plan:
    intent: str                                 # lookup | ranking | trend | comparison | diagnostic
    periods: list[Period] = field(default_factory=list)
    baseline: Optional[Period] = None           # assumed comparison period (previous month)
    has_history: bool = False
    hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def complexity(self) -> str:
        return "complex" if self.intent in ("diagnostic", "comparison") else "simple"

    @property
    def period_named(self) -> bool:
        return bool(self.periods)

    def summary(self) -> str:
        span = " vs ".join(p.label for p in ([*self.periods, self.baseline] if self.baseline else self.periods))
        return f"{self.intent} ({self.complexity})" + (f" - {span}" if span else " - whole data window")

    def render(self) -> str:
        lines = ["QUERY PLAN (prepared by the app from the question; follow it exactly):",
                 f"- Intent: {self.intent} ({self.complexity})"]
        for p in self.periods:
            lines.append(f"- Period: {p.label} -> {p.sql()}")
        if self.baseline:
            lines.append(f"- Compare against: {self.baseline.label} -> {self.baseline.sql()} "
                         "(assumed: the previous month; say so in the answer)")
        if not self.periods:
            lines.append("- Period: reuse the period from the earlier conversation if this is a follow-up."
                         if self.has_history else
                         "- Period: none named -> use the WHOLE data window and add NO order_date condition.")
        lines += [f"- Note: {n}" for n in self.notes] + [f"- Hint: {h}" for h in self.hints]
        return "\n".join(lines)


def make_plan(question: str, window: Optional[tuple[str, str]], has_history: bool = False) -> Plan:
    q = question.lower()
    data_min = date.fromisoformat(window[0]) if window else date.today()
    data_max = date.fromisoformat(window[1]) if window else date.today()
    periods = resolve_periods(question, data_max)

    if _WHY.search(q):
        intent = "diagnostic"
    elif _COMPARE.search(q) or len(periods) >= 2 or (_CHANGE.search(q) and periods):
        intent = "comparison"
    elif _TREND.search(q):
        intent = "trend"
    elif _RANK.search(q):
        intent = "ranking"
    else:
        intent = "lookup"
    plan = Plan(intent, periods, has_history=has_history)

    if intent in ("diagnostic", "comparison") and len(periods) == 1 and (periods[0].end - periods[0].start).days <= 31:
        prev_end = periods[0].start - timedelta(days=1)
        prev = month_period(prev_end.year, prev_end.month)
        if prev.start >= data_min.replace(day=1):
            plan.baseline = prev
    for p in [*periods, *([plan.baseline] if plan.baseline else [])]:
        if p.end <= data_min or p.start > data_max:
            plan.notes.append(f"{p.label} lies outside the data window ({window[0]} to {window[1]}); "
                              "expect no rows and say so.")
    if intent in ("diagnostic", "comparison"):
        plan.hints.append("Use ONE query with conditional aggregation by category (or product) returning both "
                          "periods plus the change and change_pct. Do not claim causes the data does not show.")
    if intent == "trend":
        plan.hints.append("GROUP BY substr(order_date, 1, 7) AS month and ORDER BY month.")
    if intent == "ranking":
        plan.hints.append("Sort descending and LIMIT to the number requested (default 5). Rank by completed "
                          "revenue unless the user says units or quantity.")
    if re.search(r"\bcustomers?\b", q) and (intent == "ranking" or re.search(r"\b(each|per|by|how many)\b", q)):
        plan.hints.append("Customer names are not unique: GROUP BY customers.customer_id, name only for display.")
    if "revenue" in q or "sales" in q:
        plan.hints.append("Revenue = SUM(orders.revenue) over orders with status = 'completed'.")
    return plan


# ----------------------------------------------------------------------------- pre-flight
_WRITE = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:delete|drop|truncate|erase|wipe|insert|alter|remove|update)\b"
                    r"|\b(?:delete|drop|truncate|erase|wipe)\s+(?:the\s+|all\s+|every\s+)?(?:table|database|orders?|customers?|products?|records?|rows?|data|everything)\b")
_FOLLOWUP = re.compile(r"^\s*(?:and|what about|how about|same for|same thing for|also|what of)\b")
_FILLER = {"show", "me", "give", "tell", "what", "is", "was", "are", "were", "our", "the", "my", "a", "an",
           "please", "can", "you", "could", "display", "see", "view", "get", "let", "us", "i", "want", "to",
           "know", "about", "of", "for", "do", "we", "have"}
_BARE_METRIC = {"revenue", "sales", "orders", "customers", "products", "income", "earnings"}


def preflight(question: str, has_history: bool, window: Optional[tuple[str, str]]) -> Optional[Preflight]:
    q = (question or "").strip().lower()
    if not q:
        return Preflight("clarification", "What would you like to know about your orders, customers or products?")
    if _WRITE.search(q):
        return Preflight("refused", REFUSAL)

    data_max = date.fromisoformat(window[1]) if window else date.today()
    periods = resolve_periods(question, data_max)
    span = f" ({window[0]} to {window[1]})" if window else ""

    if _FOLLOWUP.search(q) and not has_history:
        return Preflight("clarification", "I don't have an earlier question to build on. Which metric and "
                                          "period do you mean, for example 'revenue in August'?")
    if _WHY.search(q) and not periods and not has_history:
        return Preflight("clarification", "Which period should I analyse? For example 'Why did revenue "
                                          "decrease in September?'. I'll compare it with the previous month.")
    content = [w for w in re.findall(r"[a-z']+", q) if w not in _FILLER]
    if content and all(w in _BARE_METRIC for w in content) and not periods and not has_history:
        return Preflight("clarification", f"Which period do you want for {content[-1]}: a specific month or "
                                          f"quarter, or the entire available data{span}?")
    return None


# ----------------------------------------------------------------------------- abstention
_STATIC_TERMS = {"revenue", "sales", "income", "sold", "selling", "buyer", "spend", "spent", "spending", "order",
                 "customer", "product", "category", "aov", "average", "total", "count", "cancellation", "cancelled",
                 "return", "returned", "growth", "profit", "price", "pricing", "discount", "signup", "signed",
                 "city", "segment", "month", "monthly", "year", "quarter", "unit", "quantity", "completed", "pending"}


def _norm(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _tokens(text: str) -> set[str]:
    return {_norm(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}   # skip "s", "x", "of" ...


@lru_cache(maxsize=4)
def _vocabulary(db_path: str) -> frozenset[str]:  # noqa: ARG001 (db_path is the cache key)
    words = set(_STATIC_TERMS)
    conn = get_connection()
    try:
        for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            words |= _tokens(t)
            words |= {n for row in conn.execute(f'PRAGMA table_info("{t}")') for n in _tokens(row[1])}
        for sql in ("SELECT name FROM products", "SELECT DISTINCT category FROM products",
                    "SELECT DISTINCT city FROM customers", "SELECT DISTINCT segment FROM customers",
                    "SELECT DISTINCT status FROM orders"):
            try:
                for (v,) in conn.execute(sql):
                    words |= _tokens(str(v))
            except sqlite3.Error:
                pass
    finally:
        conn.close()
    return frozenset(words)


def is_out_of_scope(question: str, low_confidence: bool) -> bool:
    """Abstain only when BOTH signals agree: weak retrieval AND no word in the question matches the data
    vocabulary (tables, columns, products, cities...). Either signal alone could be a false alarm."""
    if not low_confidence:
        return False
    return not (_tokens(question) & _vocabulary(str(get_settings().db_path)))