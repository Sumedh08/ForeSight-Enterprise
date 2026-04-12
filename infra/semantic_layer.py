from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
from infra.settings import settings
from infra.nim_gateway import nim_gateway

logger = logging.getLogger(__name__)

@dataclass
class SemanticProfile:
    table_name: str
    description: str = ""
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    entities: dict[str, list[str]] = field(default_factory=dict)  # col -> [distinct values]
    
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    is_metric: bool = False
    is_dimension: bool = False
    description: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

class SemanticLayer:
    def __init__(self):
        self.profiles_path = settings.data_dir / "cache" / "semantic_profiles.json"
        self.profiles: dict[str, SemanticProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        if self.profiles_path.exists():
            try:
                data = json.loads(self.profiles_path.read_text())
                for table, p in data.items():
                    self.profiles[table] = SemanticProfile(
                        table_name=p["table_name"],
                        description=p["description"],
                        columns={c: ColumnProfile(**v) for c, v in p["columns"].items()},
                        entities=p["entities"]
                    )
            except Exception as e:
                logger.error(f"Failed to load semantic profiles: {e}")

    def _save_profiles(self):
        data = {k: {
            "table_name": v.table_name,
            "description": v.description,
            "columns": {ck: cv.__dict__ for ck, cv in v.columns.items()},
            "entities": v.entities
        } for k, v in self.profiles.items()}
        self.profiles_path.write_text(json.dumps(data, indent=2))

    async def profile_table(self, table_name: str):
        """Perform deep scan of a table to extract business meaning and entities."""
        logger.info(f"Profiling table: {table_name}")
        
        conn = duckdb.connect(str(settings.duckdb_path))
        try:
            # 1. Get Schema
            schema = conn.execute(f"DESCRIBE {table_name}").df()
            profile = SemanticProfile(table_name=table_name)
            
            for _, row in schema.iterrows():
                col_name = row["column_name"]
                dtype = str(row["column_type"])
                
                # Basic Categorization
                is_metric = any(t in dtype.upper() for t in ["DOUBLE", "FLOAT", "DECIMAL", "INTEGER", "BIGINT"])
                is_dimension = "VARCHAR" in dtype.upper()
                
                col_profile = ColumnProfile(name=col_name, dtype=dtype, is_metric=is_metric, is_dimension=is_dimension)
                
                # 2. Extract Entities for Dimensions
                if is_dimension:
                    try:
                        # Limit to top 100 unique values to keep profile manageable
                        distinct = conn.execute(f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 101").df()
                        if len(distinct) <= 100:
                            profile.entities[col_name] = distinct[col_name].dropna().astype(str).tolist()
                        else:
                            profile.entities[col_name] = ["<too many values>"]
                    except Exception:
                        pass
                
                profile.columns[col_name] = col_profile
                
            # 3. Business Context Generation via NIM
            if nim_gateway.enabled:
                context_prompt = f"Analyze this table schema and entity list. Give a 1-sentence business description of what this table represents.\nTable: {table_name}\nColumns: {list(profile.columns.keys())}\nEntities: {json.dumps(profile.entities)}"
                try:
                    desc = await nim_gateway.chat([{"role": "user", "content": context_prompt}], temperature=0.1)
                    profile.description = desc.strip()
                except Exception as e:
                    logger.warning(f"NIM failed to generate table description: {e}")

            self.profiles[table_name] = profile
            self._save_profiles()
            
        finally:
            conn.close()

    def get_context_for_sql(self) -> str:
        """Stringify profiles for LLM context."""
        ctx = "### DATASET BUSINESS CONTEXT\n"
        for p in self.profiles.values():
            ctx += f"Table '{p.table_name}': {p.description}\n"
            for col, entities in p.entities.items():
                if entities and entities[0] != "<too many values>":
                    ctx += f"  - Column '{col}' contains entities: {', '.join(entities[:10])}...\n"
        return ctx

semantic_layer = SemanticLayer()
