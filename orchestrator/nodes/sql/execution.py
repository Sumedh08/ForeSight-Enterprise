from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError

from infra.db import get_duckdb_conn


def _run_query(sql: str) -> dict:
    conn = get_duckdb_conn(read_only=True)
    try:
        result = conn.execute(sql).fetchdf()
        return {
            "row_count": int(len(result.index)),
            "preview_rows": result.head(20).to_dict("records"),
            "columns": list(result.columns),
        }
    finally:
        conn.close()


def execute_sql(sql: str, *, timeout_s: int) -> dict:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_query, sql)
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError as exc:
            raise RuntimeError(f"Query exceeded timeout of {timeout_s}s") from exc
