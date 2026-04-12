from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    name: str
    data_type: str
    sample_values: list[Any]


@dataclass(frozen=True, slots=True)
class TableSchema:
    name: str
    columns: list[ColumnSchema]


@dataclass(frozen=True, slots=True)
class DatabaseSchema:
    dialect: str
    tables: list[TableSchema]


@dataclass(frozen=True, slots=True)
class QueryExecutionResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


class DatabaseConnector(ABC):
    dialect: str

    @abstractmethod
    def introspect_schema(self, sample_limit: int = 3) -> DatabaseSchema:
        raise NotImplementedError

    @abstractmethod
    def dry_run(self, sql: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, sql: str, preview_limit: int = 50) -> QueryExecutionResult:
        raise NotImplementedError


def serialize_schema(schema: DatabaseSchema, table_names: list[str] | None = None) -> str:
    allowed = {name.lower() for name in table_names} if table_names else None
    lines = [f"Dialect: {schema.dialect}"]
    for table in schema.tables:
        if allowed is not None and table.name.lower() not in allowed:
            continue
        lines.append(f"Table {table.name}:")
        for column in table.columns:
            samples = ", ".join(str(item) for item in column.sample_values if item is not None)
            sample_suffix = f" | sample values: {samples}" if samples else ""
            lines.append(f"  - {column.name} ({column.data_type}){sample_suffix}")
    return "\n".join(lines)


class DuckDBConnector(DatabaseConnector):
    dialect = "duckdb"

    def __init__(self, database_path: str | Path, *, read_only: bool = True) -> None:
        self.database_path = str(database_path)
        self.read_only = read_only

    def _connect(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(self.database_path, read_only=self.read_only)
        conn.execute("SET threads = 4")
        conn.execute("SET memory_limit = '1GB'")
        return conn

    def introspect_schema(self, sample_limit: int = 3) -> DatabaseSchema:
        with self._connect() as conn:
            table_names = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'main'
                    ORDER BY table_name
                    """
                ).fetchall()
            ]
            tables: list[TableSchema] = []
            for table_name in table_names:
                columns = conn.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = ?
                    ORDER BY ordinal_position
                    """,
                    [table_name],
                ).fetchall()
                table_columns: list[ColumnSchema] = []
                for column_name, data_type in columns:
                    sample_rows = conn.execute(
                        f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                        f'WHERE "{column_name}" IS NOT NULL LIMIT {sample_limit}'
                    ).fetchall()
                    table_columns.append(
                        ColumnSchema(
                            name=column_name,
                            data_type=str(data_type),
                            sample_values=[row[0] for row in sample_rows],
                        )
                    )
                tables.append(TableSchema(name=table_name, columns=table_columns))
        return DatabaseSchema(dialect=self.dialect, tables=tables)

    def dry_run(self, sql: str) -> None:
        with self._connect() as conn:
            conn.execute(f"EXPLAIN {sql}").fetchall()

    def execute(self, sql: str, preview_limit: int = 50) -> QueryExecutionResult:
        with self._connect() as conn:
            frame = conn.execute(sql).fetchdf()
        preview = frame.head(preview_limit)
        return QueryExecutionResult(
            columns=list(preview.columns),
            rows=preview.to_dict("records"),
            row_count=int(len(frame.index)),
        )


class SQLiteConnector(DatabaseConnector):
    dialect = "sqlite"

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def introspect_schema(self, sample_limit: int = 3) -> DatabaseSchema:
        with self._connect() as conn:
            tables = []
            table_rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            for table_row in table_rows:
                table_name = table_row["name"]
                pragma_rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                columns: list[ColumnSchema] = []
                for column in pragma_rows:
                    column_name = column["name"]
                    sample_rows = conn.execute(
                        f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                        f'WHERE "{column_name}" IS NOT NULL LIMIT {sample_limit}'
                    ).fetchall()
                    columns.append(
                        ColumnSchema(
                            name=column_name,
                            data_type=str(column["type"]),
                            sample_values=[row[0] for row in sample_rows],
                        )
                    )
                tables.append(TableSchema(name=table_name, columns=columns))
        return DatabaseSchema(dialect=self.dialect, tables=tables)

    def dry_run(self, sql: str) -> None:
        with self._connect() as conn:
            conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()

    def execute(self, sql: str, preview_limit: int = 50) -> QueryExecutionResult:
        with self._connect() as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description or []]
        preview_rows = [dict(row) for row in rows[:preview_limit]]
        return QueryExecutionResult(columns=columns, rows=preview_rows, row_count=len(rows))


class SQLAlchemyConnector(DatabaseConnector):
    def __init__(self, *, dialect: str, dsn: str, read_only: bool = True) -> None:
        self.dialect = dialect
        self.dsn = dsn
        self.read_only = read_only

    def _engine(self) -> Engine:
        return create_engine(self.dsn, future=True)

    def _readonly_sql(self) -> str:
        if self.dialect == "postgres":
            return "SET TRANSACTION READ ONLY"
        if self.dialect == "mysql":
            return "SET TRANSACTION READ ONLY"
        return ""

    def _quote_identifier(self, identifier: str) -> str:
        if self.dialect == "mysql":
            return f"`{identifier}`"
        return f'"{identifier}"'

    def introspect_schema(self, sample_limit: int = 3) -> DatabaseSchema:
        with self._engine().connect() as conn:
            if self.dialect == "postgres":
                table_rows = conn.execute(
                    text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        ORDER BY table_name
                        """
                    )
                ).fetchall()
            else:
                table_rows = conn.execute(
                    text(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = DATABASE()
                        ORDER BY table_name
                        """
                    )
                ).fetchall()

            tables: list[TableSchema] = []
            for row in table_rows:
                table_name = str(row[0])
                column_rows = conn.execute(
                    text(
                        """
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = :table_name
                        ORDER BY ordinal_position
                        """
                    ),
                    {"table_name": table_name},
                ).fetchall()
                table_columns: list[ColumnSchema] = []
                for column_name, data_type in column_rows:
                    quoted_table = self._quote_identifier(table_name)
                    quoted_column = self._quote_identifier(str(column_name))
                    try:
                        sample_rows = conn.execute(
                            text(
                                f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
                                "WHERE "
                                f"{quoted_column} IS NOT NULL LIMIT :sample_limit"
                            ),
                            {"sample_limit": sample_limit},
                        ).fetchall()
                        sample_values = [sample[0] for sample in sample_rows]
                    except Exception:
                        sample_values = []
                    table_columns.append(
                        ColumnSchema(
                            name=str(column_name),
                            data_type=str(data_type),
                            sample_values=sample_values,
                        )
                    )
                tables.append(TableSchema(name=table_name, columns=table_columns))
        return DatabaseSchema(dialect=self.dialect, tables=tables)

    def dry_run(self, sql: str) -> None:
        with self._engine().connect() as conn:
            with conn.begin():
                readonly_sql = self._readonly_sql()
                if self.read_only and readonly_sql:
                    conn.execute(text(readonly_sql))
                explain_prefix = "EXPLAIN"
                conn.execute(text(f"{explain_prefix} {sql}")).fetchall()

    def execute(self, sql: str, preview_limit: int = 50) -> QueryExecutionResult:
        with self._engine().connect() as conn:
            with conn.begin():
                readonly_sql = self._readonly_sql()
                if self.read_only and readonly_sql:
                    conn.execute(text(readonly_sql))
                result = conn.execute(text(sql))
                rows = [dict(row._mapping) for row in result.fetchall()]
                columns = list(rows[0].keys()) if rows else list(result.keys())
        return QueryExecutionResult(columns=columns, rows=rows[:preview_limit], row_count=len(rows))


class PostgresConnector(SQLAlchemyConnector):
    dialect = "postgres"

    def __init__(self, dsn: str, *, read_only: bool = True) -> None:
        super().__init__(dialect="postgres", dsn=dsn, read_only=read_only)


class MySQLConnector(SQLAlchemyConnector):
    dialect = "mysql"

    def __init__(self, dsn: str, *, read_only: bool = True) -> None:
        super().__init__(dialect="mysql", dsn=dsn, read_only=read_only)
