from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd
from sqlalchemy import create_engine

from infra.connection_profiles import ConnectionProfileManager


def sanitize_table_name(name: str) -> str:
    table = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()
    return table or "uploaded_table"


def write_dataframe_to_profile(
    *,
    profile: dict[str, Any],
    frame: pd.DataFrame,
    table_name: str,
    profile_manager: ConnectionProfileManager,
) -> str:
    normalized_name = sanitize_table_name(table_name)
    connection_type = str(profile.get("type"))
    config = profile.get("config", {})

    if connection_type == "duckdb":
        conn = duckdb.connect(config["path"], read_only=False)
        try:
            conn.register("upload_df", frame)
            conn.execute(f'CREATE OR REPLACE TABLE "{normalized_name}" AS SELECT * FROM upload_df')
        finally:
            conn.close()
        return normalized_name

    if connection_type == "sqlite":
        engine = create_engine(f"sqlite:///{config['path']}")
        with engine.begin() as transaction:
            frame.to_sql(normalized_name, transaction, if_exists="replace", index=False)
        return normalized_name

    if connection_type in {"postgres", "mysql"}:
        dsn = profile_manager.profile_to_dsn(profile)
        engine = create_engine(dsn)
        with engine.begin() as transaction:
            frame.to_sql(normalized_name, transaction, if_exists="replace", index=False)
        return normalized_name

    raise ValueError(f"Unsupported profile type '{connection_type}'.")
