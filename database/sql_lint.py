"""Business-rule lint for SQL the model wrote (separate from safety validation).

Safety validation asks "is this query allowed?"; lint asks "does it respect our metric definitions?".
The agent feeds lint errors back to the model for a corrected query."""
from dataclasses import dataclass
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


@dataclass
class LintContext:
    window_min: Optional[str] = None   # 'YYYY-MM-DD' first order date
    window_max: Optional[str] = None   # 'YYYY-MM-DD' last order date
    # True when the question named no period (and is not a follow-up)
    check_period: bool = False


@dataclass
class LintIssue:
    rule: str
    message: str
    severity: str = "error"


_FLIP = {exp.GTE: exp.LTE, exp.GT: exp.LT, exp.LTE: exp.GTE, exp.LT: exp.GT}


def _alias_map(tree) -> dict[str, str]:
    return {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}


def _group_exprs(tree) -> list:
    group = tree.args.get("group") if isinstance(tree, exp.Select) else None
    if not group:
        return []
    items = tree.expressions
    aliases = {e.alias.lower(): e.this for e in items if isinstance(e, exp.Alias)}
    out = []
    for g in group.expressions:
        if isinstance(g, exp.Literal) and not g.is_string and g.name.isdigit() and 0 < int(g.name) <= len(items):
            g = items[int(g.name) - 1]
            g = g.this if isinstance(g, exp.Alias) else g
        elif isinstance(g, exp.Column) and not g.table and g.name.lower() in aliases:
            g = aliases[g.name.lower()]
        out.append(g)
    return out


def _is_col(node, name: str) -> bool:
    return isinstance(node, exp.Column) and node.name.lower() == name


def _customer_name_grouping(tree) -> bool:
    cols = [g for g in _group_exprs(tree) if isinstance(g, exp.Column)]
    amap = _alias_map(tree)
    if any(_is_col(c, "customer_id") for c in cols):
        return False
    for c in cols:
        if _is_col(c, "name"):
            table = amap.get(c.table.lower()) if c.table else ("customers" if "customers" in amap.values()
                                                               and "products" not in amap.values() else None)
            if table == "customers":
                return True
    return False


def _revenue_without_status(tree) -> bool:
    sums = [a for a in tree.find_all(exp.Sum, exp.Avg) if any(
        _is_col(c, "revenue") for c in a.find_all(exp.Column))]
    return bool(sums) and not any(_is_col(c, "status") for c in tree.find_all(exp.Column))


def _narrows_period(tree, ctx: LintContext) -> bool:
    for node in tree.find_all(exp.GTE, exp.GT, exp.LT, exp.LTE, exp.EQ, exp.NEQ, exp.Between, exp.In, exp.Like):
        if not any(_is_col(c, "order_date") for c in node.find_all(exp.Column)):
            continue
        cmp_type = type(node)
        if cmp_type in _FLIP and ctx.window_min and ctx.window_max:
            left, right = node.this, node.expression
            if _is_col(right, "order_date") and isinstance(left, exp.Literal) and left.is_string:
                left, right, cmp_type = right, left, _FLIP[cmp_type]
            if _is_col(left, "order_date") and isinstance(right, exp.Literal) and right.is_string:
                v = right.name
                harmless = ((cmp_type is exp.GTE and v <= ctx.window_min) or (cmp_type is exp.GT and v < ctx.window_min)
                            or (cmp_type is exp.LT and v > ctx.window_max) or (cmp_type is exp.LTE and v >= ctx.window_max))
                if harmless:
                    continue
        return True
    return False


def lint_sql(sql: str, ctx: Optional[LintContext] = None) -> list[LintIssue]:
    ctx = ctx or LintContext()
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return []                          # unparseable SQL is the validator's job
    issues: list[LintIssue] = []
    if _customer_name_grouping(tree):
        issues.append(LintIssue("customer_name_grouping",
                                "Customer names are not unique, so grouping by name merges different customers. "
                                "GROUP BY customers.customer_id (and name for display)."))
    if _revenue_without_status(tree):
        issues.append(LintIssue("revenue_needs_completed",
                                "Revenue must only count completed orders: add orders.status = 'completed'."))
    if ctx.check_period and _narrows_period(tree, ctx):
        issues.append(LintIssue("unrequested_period",
                                "The question named no period, so do not filter on order_date. Use the whole "
                                "data window."))
    return issues
