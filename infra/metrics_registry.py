from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from infra.settings import settings


@dataclass(slots=True)
class MetricDefinition:
    key: str
    label: str
    description: str
    aliases: list[str]
    default_grain: str
    allowed_grains: list[str]
    default_horizon: int
    max_horizon: int
    dimensions: list[str]
    sql_template: str


class MetricRegistry:
    """
    Fully dynamic metric registry.
    
    1. Reads optional YAML config (for pre-defined metrics)
    2. Auto-registers ALL numeric/castable columns from the schema cache
    3. LLM-driven resolution maps any natural-language question to the best metric
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.metric_registry_path
        self.metrics: dict[str, MetricDefinition] = {}

        # Load from YAML config (if any pre-defined metrics exist)
        try:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            for key, value in (payload.get("metrics") or {}).items():
                self.metrics[key] = MetricDefinition(
                    key=key,
                    label=value["label"],
                    description=value["description"],
                    aliases=list(value.get("aliases", [])),
                    default_grain=value["default_grain"],
                    allowed_grains=list(value.get("allowed_grains", [value["default_grain"]])),
                    default_horizon=int(value.get("default_horizon", 4)),
                    max_horizon=int(value.get("max_horizon", 8)),
                    dimensions=list(value.get("dimensions", [])),
                    sql_template=value["sql_template"],
                )
        except Exception:
            pass

        # Auto-register from schema cache — works for ANY uploaded dataset
        self._auto_register_from_schema()

    def _auto_register_from_schema(self) -> None:
        """Scan schema cache and register every forecastable column automatically."""
        if not settings.schema_cache_path.exists():
            return
        try:
            schema_cache = json.loads(settings.schema_cache_path.read_text(encoding="utf-8"))
            details = schema_cache.get("details", {})
            for table_name, table_info in details.items():
                col_list = table_info.get("columns", [])
                col_map = {c["name"]: c["type"] for c in col_list if isinstance(c, dict)}

                # Find a date/time column by type first, then by name
                date_col = next(
                    (name for name, dtype in col_map.items()
                     if any(k in dtype.lower() for k in ("date", "time", "timestamp"))),
                    None,
                )
                if not date_col:
                    date_col = next(
                        (name for name in col_map
                         if any(k in name.lower() for k in ("date", "time", "month", "year", "day", "period", "week"))),
                        None,
                    )

                if not date_col:
                    continue

                # Register all numeric and potentially-numeric columns
                NUMERIC_TYPES = ("int", "bigint", "double", "float", "numeric", "real", "decimal", "hugeint", "smallint", "tinyint")
                SKIP_NAMES = ("name", "id", "url", "type", "category", "symbol", "slug", "rank")

                for col, dtype in col_map.items():
                    if col == date_col:
                        continue

                    is_numeric = any(k in dtype.lower() for k in NUMERIC_TYPES)
                    is_varchar = "varchar" in dtype.lower()

                    if not is_numeric and not is_varchar:
                        continue

                    # Skip VARCHARs that are clearly identifiers, not values
                    if is_varchar and any(k in col.lower() for k in SKIP_NAMES):
                        continue

                    metric_key = f"{table_name}_{col}"
                    if metric_key in self.metrics:
                        continue

                    # Build human-readable label
                    label = f"{table_name} {col}".replace("_", " ").title()

                    # Build aliases — include the column name, table name, and natural variants
                    aliases = list(set([
                        col,
                        col.replace("_", " "),
                        table_name,
                        table_name.replace("_", " "),
                        f"{table_name} {col}",
                        f"{col} {table_name}",
                        label.lower(),
                    ]))

                    # For VARCHAR columns, strip currency/percent symbols before CAST
                    if is_varchar:
                        value_expr = f"TRY_CAST(REGEXP_REPLACE({col}, '[^0-9.\\-]', '', 'g') AS DOUBLE)"
                    else:
                        value_expr = col

                    self.metrics[metric_key] = MetricDefinition(
                        key=metric_key,
                        label=label,
                        description=f"Auto-registered: {col} from {table_name}",
                        aliases=aliases,
                        default_grain="day",
                        allowed_grains=["day", "week", "month", "quarter", "year"],
                        default_horizon=4,
                        max_horizon=24,
                        dimensions=[],
                        sql_template=(
                            f"SELECT {date_col} AS period, {value_expr} AS value "
                            f"FROM {table_name} "
                            f"WHERE 1=1 {{filter_clause}} "
                            f"AND {date_col} IS NOT NULL AND {col} IS NOT NULL "
                            f"ORDER BY {date_col}"
                        ),
                    )

        except Exception:
            pass

    def known_terms(self) -> set[str]:
        terms: set[str] = set()
        for metric in self.metrics.values():
            terms.add(metric.key.lower())
            terms.add(metric.label.lower())
            terms.update(alias.lower() for alias in metric.aliases)
        return terms

    def resolve(
        self,
        *,
        question: str,
        metric: str | None = None,
        series_id: str | None = None,
        grain: str | None = None,
    ) -> MetricDefinition | None:
        """Keyword-based resolution (synchronous fallback)."""
        if metric and metric in self.metrics:
            candidate = self.metrics[metric]
            return candidate if not grain or grain in candidate.allowed_grains else None
        if series_id and series_id in self.metrics:
            candidate = self.metrics[series_id]
            return candidate if not grain or grain in candidate.allowed_grains else None

        lowered = question.lower()
        ranked: list[tuple[int, MetricDefinition]] = []
        for definition in self.metrics.values():
            score = 0
            for term in [definition.key, definition.label, *definition.aliases]:
                if term.lower() in lowered:
                    score += max(1, len(term.split()))
            if score:
                ranked.append((score, definition))

        if not ranked:
            # If no keyword match, return the first metric as a last resort
            # (better than returning None when there IS data available)
            if self.metrics:
                return next(iter(self.metrics.values()))
            return None

        ranked.sort(key=lambda item: item[0], reverse=True)
        candidate = ranked[0][1]
        if grain and grain not in candidate.allowed_grains:
            return None
        return candidate

    async def resolve_with_llm(
        self,
        *,
        question: str,
        metric: str | None = None,
        grain: str | None = None,
    ) -> tuple[MetricDefinition | None, dict[str, str] | None]:
        """
        LLM-driven metric resolution.
        
        Returns (metric_definition, filter_dict) where filter_dict contains
        any WHERE clause conditions the LLM extracted (e.g., {"name": "Bitcoin"}).
        """
        # Quick check — if explicit metric ID was given, use it
        if metric and metric in self.metrics:
            candidate = self.metrics[metric]
            if not grain or grain in candidate.allowed_grains:
                return candidate, None
            return None, None

        if not self.metrics:
            return None, None

        # Build metric catalog for the LLM
        from infra.nim_gateway import nim_gateway
        if not nim_gateway.enabled:
            return self.resolve(question=question, metric=metric, grain=grain), None

        catalog = []
        for m in self.metrics.values():
            catalog.append(f"- key: {m.key} | label: {m.label} | aliases: {', '.join(m.aliases[:5])}")

        # Read schema for WHERE clause context
        schema_context = ""
        try:
            schema_cache = json.loads(settings.schema_cache_path.read_text(encoding="utf-8"))
            for table_name, info in schema_cache.get("details", {}).items():
                cols = info.get("columns", [])
                samples = info.get("sample_values", {})
                col_desc = [f"{c['name']} ({c['type']})" for c in cols[:10]]
                schema_context += f"\nTable {table_name}: {', '.join(col_desc)}"
                for col_name, vals in list(samples.items())[:5]:
                    schema_context += f"\n  {col_name} samples: {vals[:3]}"
        except Exception:
            pass

        # Get deep semantic context (entities, business meanings)
        from infra.semantic_layer import semantic_layer
        semantic_context = semantic_layer.get_context_for_sql()

        prompt = f"""You are a metric resolution engine. Given a user question, pick the BEST metric key and identify any WHERE clause filters.

Available metrics:
{chr(10).join(catalog)}

Semantic Discovery Results (Discovered Entities & Business Meanings):
{semantic_context}

Schema context:
{schema_context}

User question: {question}

Respond in this exact JSON format (no markdown):
{{"metric_key": "<the best metric key>", "filters": {{"column_name": "value"}}}}

Rules:
- "metric_key" MUST be one of the keys listed above.
- "filters" should capture entity filters (e.g., if user asks about "Bitcoin", add {{"name": "Bitcoin"}}).
- CRITICAL: DO NOT add dates, years, or timeframes to the "filters" object. Dates are handled externally as target horizons.
- If no filter is needed, use an empty object: {{}}.
- Respond with ONLY the JSON. No explanation."""

        try:
            response = await nim_gateway.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            # Parse LLM response
            text = response.strip()
            # Remove markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)
            metric_key = result.get("metric_key", "")
            filters = result.get("filters", {})

            if metric_key in self.metrics:
                candidate = self.metrics[metric_key]
                if grain and grain not in candidate.allowed_grains:
                    return None, None
                return candidate, filters if filters else None

        except Exception:
            pass

        # Fallback to keyword resolution
        return self.resolve(question=question, metric=metric, grain=grain), None

    def render_sql(self, definition: MetricDefinition, filters: dict[str, Any] | None = None) -> str:
        """Render the SQL template with dynamic filters. No hardcoded column names."""
        filters = filters or {}

        # Build WHERE clause from dimension filters
        filter_parts = []
        for col_name, col_value in filters.items():
            if col_name in ("metric", "grain", "horizon"):
                continue
            safe_val = str(col_value).replace("'", "''")
            filter_parts.append(f"AND {col_name} = '{safe_val}'")

        filter_clause = " ".join(filter_parts)

        # Support both old {region_filter} and new {filter_clause} placeholders
        sql = definition.sql_template
        if "{filter_clause}" in sql:
            sql = sql.format(filter_clause=filter_clause)
        elif "{region_filter}" in sql:
            # Legacy support
            region = filters.get("region", "")
            region_filter = f" AND region = '{region}'" if region else ""
            sql = sql.format(region_filter=region_filter)

        return sql
