from __future__ import annotations

from infra.db import get_duckdb_conn
from infra.settings import settings
from safety.sql_guard import guard_sql


def dry_run(sql: str) -> None:
    conn = get_duckdb_conn(read_only=True)
    try:
        conn.execute(f"EXPLAIN {sql}").fetchall()
    finally:
        conn.close()


def validate_sql_candidate(candidate_sql: str, *, allowed_tables: set[str], dialect: str = "duckdb") -> dict:
    guarded = guard_sql(
        candidate_sql,
        dialect=dialect,
        allowed_tables=allowed_tables,
        row_cap=settings.default_row_cap,
    )
    dry_run(guarded.sql)
    return {"status": "valid", "sql": guarded.sql, "tables": guarded.tables}
