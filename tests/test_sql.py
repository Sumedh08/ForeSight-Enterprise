from __future__ import annotations

import pytest

from safety.sql_guard import guard_sql


def test_guard_blocks_multi_statement():
    with pytest.raises(ValueError):
        guard_sql(
            "SELECT * FROM sales; DROP TABLE sales;",
            dialect="duckdb",
            allowed_tables={"sales"},
            row_cap=500,
        )


def test_guard_blocks_non_allowlisted_table():
    with pytest.raises(ValueError):
        guard_sql(
            "SELECT * FROM secret_table",
            dialect="duckdb",
            allowed_tables={"sales"},
            row_cap=500,
        )


def test_guard_injects_limit_when_missing():
    guarded = guard_sql(
        "SELECT * FROM sales",
        dialect="duckdb",
        allowed_tables={"sales"},
        row_cap=25,
    )
    assert "LIMIT 25" in guarded.sql.upper()
