from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


FORBIDDEN_NODES = tuple(
    node
    for node in (
        getattr(exp, "Insert", None),
        getattr(exp, "Update", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Create", None),
        getattr(exp, "Drop", None),
        getattr(exp, "Alter", None),
        getattr(exp, "Command", None),
        getattr(exp, "Merge", None),
        getattr(exp, "Copy", None),
    )
    if node is not None
)


@dataclass(slots=True)
class GuardResult:
    sql: str
    parsed: exp.Expression
    tables: list[str]


def parse_single_statement(sql: str, dialect: str) -> exp.Expression:
    statements = sqlglot.parse(sql, read=dialect)
    if len(statements) != 1:
        raise ValueError("Only a single SQL statement is allowed.")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ValueError("Only read-only query statements are allowed.")
    if any(statement.find(node_type) is not None for node_type in FORBIDDEN_NODES):
        raise ValueError("Write or DDL operations are blocked.")
    return statement


def extract_tables(statement: exp.Expression) -> list[str]:
    names = []
    for table in statement.find_all(exp.Table):
        if table.name and table.name not in names:
            names.append(table.name)
    return names


def validate_allowlist(statement: exp.Expression, allowed_tables: set[str]) -> list[str]:
    allowed = {name.lower() for name in allowed_tables}
    referenced = {name.lower() for name in extract_tables(statement)}
    return sorted(referenced - allowed)


def _get_limit_value(statement: exp.Expression) -> int | None:
    limit_expr = statement.args.get("limit")
    if not isinstance(limit_expr, exp.Limit):
        return None
    expression = limit_expr.expression
    if isinstance(expression, exp.Literal) and expression.is_int:
        return int(expression.this)
    return None


def enforce_row_cap(statement: exp.Expression, cap: int) -> exp.Expression:
    current_limit = _get_limit_value(statement)
    if current_limit is None or current_limit > cap:
        try:
            return statement.limit(cap, copy=False)
        except TypeError:  # pragma: no cover - compatibility across sqlglot versions
            return statement.limit(cap)
    return statement


def guard_sql(
    sql: str,
    *,
    dialect: str,
    allowed_tables: set[str],
    row_cap: int,
) -> GuardResult:
    statement = parse_single_statement(sql, dialect=dialect)
    blocked_tables = validate_allowlist(statement, allowed_tables)
    if blocked_tables:
        raise ValueError(f"Blocked table reference(s): {', '.join(blocked_tables)}")
    statement = enforce_row_cap(statement, cap=row_cap)
    return GuardResult(
        sql=statement.sql(dialect=dialect),
        parsed=statement,
        tables=extract_tables(statement),
    )
