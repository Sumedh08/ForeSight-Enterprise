from __future__ import annotations

import json

from infra.db import introspect_schema
from infra.settings import settings


def refresh_schema_cache() -> dict:
    schema = introspect_schema()
    settings.schema_cache_path.parent.mkdir(parents=True, exist_ok=True)
    settings.schema_cache_path.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")
    return schema


if __name__ == "__main__":
    refresh_schema_cache()
