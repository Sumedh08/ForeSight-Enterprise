"""
Dynamic MindsDB Predictor Orchestrator
Creates/updates datasource and trains predictors using active connection profile metadata.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from sqlalchemy import create_engine, text


MINDSDB_SQL_ENDPOINT = os.getenv("MINDSDB_API_URL", "http://mindsdb:47334/api/sql").rstrip("/")


def _safe_identifier(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()


def _query(sql: str) -> dict[str, Any]:
    response = requests.post(MINDSDB_SQL_ENDPOINT, json={"query": sql}, timeout=30)
    response.raise_for_status()
    body = response.json()
    if isinstance(body, dict):
        return body
    return {"data": body}


def _profile_to_dsn(profile: dict[str, Any]) -> str:
    connection_type = str(profile.get("type"))
    config = profile.get("config", {})
    if config.get("dsn"):
        return str(config["dsn"])
    if connection_type == "postgres":
        return (
            f"postgresql+psycopg2://{config.get('username')}:{config.get('password')}"
            f"@{config.get('host')}:{config.get('port', 5432)}/{config.get('database')}"
        )
    if connection_type == "mysql":
        return (
            f"mysql+pymysql://{config.get('username')}:{config.get('password')}"
            f"@{config.get('host')}:{config.get('port', 3306)}/{config.get('database')}"
        )
    raise RuntimeError("MindsDB sync currently supports postgres and mysql profiles.")


def setup_mindsdb_datasource(profile: dict[str, Any]) -> str:
    datasource_name = f"profile_{_safe_identifier(str(profile.get('id', 'default')))}"
    dsn = _profile_to_dsn(profile)
    dsn = dsn.replace("\\", "\\\\").replace('"', '\\"')
    engine = "postgres" if profile.get("type") == "postgres" else "mysql"
    sql = (
        f"CREATE DATABASE IF NOT EXISTS {datasource_name} "
        f"WITH ENGINE = '{engine}', PARAMETERS = {{\"connection\": \"{dsn}\"}};"
    )
    _query(sql)
    return datasource_name


def _discover_tables(dsn: str, requested_tables: list[str] | None = None) -> list[str]:
    engine = create_engine(dsn)
    with engine.connect() as connection:
        tables = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
            )
        ).fetchall()
        if not tables:
            tables = connection.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    ORDER BY table_name
                    """
                )
            ).fetchall()
    names = [_safe_identifier(str(row[0])) for row in tables]
    if requested_tables:
        allowed = {_safe_identifier(item) for item in requested_tables}
        names = [name for name in names if name in allowed]
    return names


def _find_target_columns(dsn: str, table_name: str) -> tuple[str | None, str | None]:
    engine = create_engine(dsn)
    with engine.connect() as connection:
        rows = connection.execute(
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

    date_column = None
    value_column = None
    for column_name, data_type in rows:
        name = str(column_name)
        dtype = str(data_type).lower()
        if date_column is None and ("date" in dtype or "time" in dtype or "timestamp" in dtype):
            date_column = name
        if value_column is None and any(token in dtype for token in ("int", "double", "float", "decimal", "numeric", "real")):
            if name.lower() not in {"id", "index"}:
                value_column = name
    return date_column, value_column


def discover_and_train(profile: dict[str, Any], *, tables: list[str] | None = None) -> dict[str, Any]:
    dsn = _profile_to_dsn(profile)
    datasource = setup_mindsdb_datasource(profile)
    discovered = _discover_tables(dsn, requested_tables=tables)
    created: list[str] = []

    for table in discovered:
        if table.endswith("_forecast"):
            continue
        date_col, value_col = _find_target_columns(dsn, table)
        if not date_col or not value_col:
            continue
        predictor_name = f"{table}_predictor"
        check = _query(f"SELECT name FROM mindsdb.predictors WHERE name = '{predictor_name}'")
        if check.get("data"):
            continue

        sql = f"""
        CREATE PREDICTOR mindsdb.{predictor_name}
        FROM {datasource}
          (SELECT * FROM {table})
        PREDICT {value_col}
        ORDER BY {date_col}
        WINDOW 20
        HORIZON 30;
        """
        _query(sql)
        created.append(predictor_name)

    return {"datasource": datasource, "predictors_created": created}


def run_from_conf(connection_profile: dict[str, Any] | str | None, tables: list[str] | None = None) -> dict[str, Any]:
    if connection_profile is None:
        raise RuntimeError("connection_profile is required for MindsDB dynamic setup.")
    if isinstance(connection_profile, str):
        profile = json.loads(connection_profile)
    else:
        profile = connection_profile
    return discover_and_train(profile, tables=tables)


if __name__ == "__main__":
    raw = os.getenv("CONNECTION_PROFILE_JSON")
    raw_tables = os.getenv("TABLES_JSON")
    table_list = json.loads(raw_tables) if raw_tables else None
    result = run_from_conf(raw, tables=table_list)
    print(json.dumps(result, indent=2))
