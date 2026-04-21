from __future__ import annotations

import re
from typing import Any

import duckdb
from sqlalchemy import create_engine

from components.connectors import DuckDBConnector, SQLiteConnector
from infra.connection_profiles import ConnectionProfileManager


def sanitize_table_name(name: str) -> str:
    table = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return table or "uploaded_table"


def write_dataframe_to_profile(
    *,
    profile: dict[str, Any],
    frame: Any,
    table_name: str,
    profile_manager: ConnectionProfileManager,
) -> str:
    normalized_name = sanitize_table_name(table_name)
    connection_type = str(profile.get("type"))
    config = profile.get("config", {})

    if connection_type == "duckdb":
        connector = profile_manager.build_connector(profile, read_only=False)
        db_path = config.get("path")
        if isinstance(connector, DuckDBConnector):
            db_path = connector.database_path
        conn = duckdb.connect(str(db_path), read_only=False)
        try:
            conn.register("upload_df", frame)
            conn.execute(f'CREATE OR REPLACE TABLE "{normalized_name}" AS SELECT * FROM upload_df')
        finally:
            conn.close()
        return normalized_name

    if connection_type == "sqlite":
        connector = profile_manager.build_connector(profile, read_only=False)
        db_path = config.get("path")
        if isinstance(connector, SQLiteConnector):
            db_path = connector.database_path
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as transaction:
            frame.to_sql(normalized_name, transaction, if_exists="replace", index=False)
        return normalized_name

    if connection_type in {"postgres", "mysql"}:
        dsn = profile_manager.profile_to_dsn(profile)
        # Harden connection with timeouts and pre-ping to handle transient Docker network issues
        engine_options = {
            "pool_pre_ping": True,
            "pool_recycle": 300,
        }
        if "postgres" in str(dsn).lower():
            engine_options["connect_args"] = {"connect_timeout": 10}

        engine = create_engine(dsn, **engine_options)
        with engine.begin() as transaction:
            frame.to_sql(normalized_name, transaction, if_exists="replace", index=False)
        return normalized_name

    raise ValueError(f"Unsupported profile type '{connection_type}'.")
