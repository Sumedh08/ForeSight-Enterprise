from __future__ import annotations

import json
from typing import Any

from infra.connection_profiles import ConnectionProfileManager
from infra.settings import settings


def refresh_schema_cache(
    *,
    profile: dict[str, Any] | None = None,
    profile_manager: ConnectionProfileManager | None = None,
) -> dict[str, Any]:
    manager = profile_manager or ConnectionProfileManager()
    active_profile = profile or manager.get_active_profile()
    connector = manager.build_connector(active_profile, read_only=True)
    schema = connector.introspect_schema(sample_limit=3)
    payload = {
        "dialect": schema.dialect,
        "tables": [table.name for table in schema.tables],
        "details": {
            table.name: {
                "columns": [{"name": column.name, "type": column.data_type} for column in table.columns],
                "sample_values": {column.name: list(column.sample_values) for column in table.columns},
            }
            for table in schema.tables
        },
    }
    settings.schema_cache_path.parent.mkdir(parents=True, exist_ok=True)
    settings.schema_cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


if __name__ == "__main__":
    refresh_schema_cache()
