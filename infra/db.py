from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from infra.settings import settings


def get_duckdb_conn(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    settings.ensure_directories()
    conn = duckdb.connect(str(settings.duckdb_path), read_only=read_only)
    conn.execute("SET threads = 4")
    conn.execute("SET memory_limit = '1GB'")
    return conn


@contextmanager
def duckdb_connection(read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    conn = get_duckdb_conn(read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def introspect_schema() -> dict:
    with duckdb_connection(read_only=True) as conn:
        tables = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchall()
        schema: dict[str, dict] = {"tables": [], "details": {}}
        for (table_name,) in tables:
            schema["tables"].append(table_name)
            columns = conn.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
                """,
                [table_name],
            ).fetchall()
            samples = {}
            for column_name, _ in columns:
                rows = conn.execute(
                    f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 3"
                ).fetchall()
                samples[column_name] = [row[0] for row in rows]
            schema["details"][table_name] = {
                "columns": [{"name": name, "type": dtype} for name, dtype in columns],
                "sample_values": samples,
            }
        return schema


async def get_pg_conn():
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN is not configured")
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("asyncpg is required for Postgres support") from exc

    return await asyncpg.connect(
        settings.postgres_dsn,
        command_timeout=settings.query_timeout_s,
        server_settings={"default_transaction_read_only": "on"},
    )


def postgres_enabled() -> bool:
    return bool(settings.postgres_dsn and os.getenv("POSTGRES_DSN"))
