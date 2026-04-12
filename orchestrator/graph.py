from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from api.models.schemas import ForecastArtifact, QueryRequest, QueryResponse, ScenarioArtifact, SQLArtifact, WarningItem
from infra.db import duckdb_connection
from infra.metrics import forecast_confidence, sql_confidence, warning
from infra.metrics_registry import MetricRegistry
from infra.nim_gateway import nim_gateway
from infra.settings import settings
from infra.vector_store import RetrievalStore
from offline.schema_refresh import refresh_schema_cache
from orchestrator.nodes.forecast.anomaly import detect_anomalies
from orchestrator.nodes.forecast.backtest import rolling_backtest
from orchestrator.nodes.forecast.baseline import seasonal_naive_forecast
from orchestrator.nodes.forecast.curator import curate_series
from orchestrator.nodes.forecast.model import run_forecast_model
from orchestrator.nodes.sql.execution import execute_sql
from orchestrator.nodes.sql.generation import generate_sql_candidates
from orchestrator.nodes.sql.repair import repair_candidate
from orchestrator.nodes.sql.retrieval import build_sql_context
from orchestrator.nodes.sql.selection import select_best_candidate
from orchestrator.nodes.sql.validation import validate_sql_candidate
from orchestrator.router import route_question

try:  # pragma: no cover - optional dependency
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    END = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


def _warning_items(items: list[dict[str, str]]) -> list[WarningItem]:
    return [WarningItem(kind=item["kind"], message=item["message"]) for item in items]


@dataclass
class AppServices:
    metric_registry: MetricRegistry
    retrieval_store: RetrievalStore
    schema_cache: dict[str, Any]
    allowed_tables: set[str]

    def refresh_schema(self) -> None:
        """Reload schema and metrics from disk cache after an upload."""
        from offline.schema_refresh import refresh_schema_cache
        refresh_schema_cache()
        self.schema_cache = json.loads(settings.schema_cache_path.read_text(encoding="utf-8"))
        self.allowed_tables = set(self.schema_cache.get("details", {}).keys()) | set(self.schema_cache.get("tables", []))
        self.metric_registry = MetricRegistry()
        self.retrieval_store = RetrievalStore()
        # Auto-generate examples from current schema
        self._generate_dynamic_examples()

    def _generate_dynamic_examples(self) -> None:
        """Auto-generate SQL examples from the current schema for retrieval context."""
        examples = []
        idx = 0
        for table_name, detail in self.schema_cache.get("details", {}).items():
            cols = [c["name"] for c in detail.get("columns", []) if isinstance(c, dict)]
            if not cols:
                continue
            # Basic select example
            idx += 1
            examples.append({
                "id": f"auto_{idx}",
                "question": f"Show all data from {table_name}",
                "sql": f"SELECT * FROM {table_name} LIMIT 20",
            })
            # Aggregation if numeric columns exist
            numeric_cols = [
                c["name"] for c in detail.get("columns", [])
                if isinstance(c, dict) and any(
                    k in c.get("type", "").lower()
                    for k in ("int", "double", "float", "numeric", "decimal")
                )
            ]
            if numeric_cols:
                col = numeric_cols[0]
                idx += 1
                examples.append({
                    "id": f"auto_{idx}",
                    "question": f"What is the average {col} in {table_name}?",
                    "sql": f"SELECT AVG({col}) AS avg_{col} FROM {table_name}",
                })
        # Write to examples.json
        import json as _json
        settings.examples_path.write_text(_json.dumps(examples, indent=2), encoding="utf-8")

    @classmethod
    def bootstrap(cls) -> "AppServices":
        settings.ensure_directories()
        if not settings.schema_cache_path.exists():
            refresh_schema_cache()
        schema_cache = json.loads(settings.schema_cache_path.read_text(encoding="utf-8"))
        return cls(
            metric_registry=MetricRegistry(),
            retrieval_store=RetrievalStore(),
            schema_cache=schema_cache,
            allowed_tables=set(schema_cache.get("details", {}).keys()) | set(schema_cache.get("tables", [])),
        )


class AnalyticsOrchestrator:
    def __init__(self, services: AppServices) -> None:
        self.services = services
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    async def run_query(self, request: QueryRequest) -> QueryResponse:
        if self.graph is not None:
            result = await self.graph.ainvoke({"request": request.model_dump(), "latency_ms": {}})
            return QueryResponse.model_validate(result["response"])

        started = time.perf_counter()
        route = await route_question(
            question=request.question,
            mode=request.mode,
            metric_registry=self.services.metric_registry,
            table_names=self.services.allowed_tables,
        )

        if route == "unclear":
            warnings = [warning("data_quality", "The request could not be confidently routed to SQL or forecasting.")]
            return QueryResponse(
                status="blocked",
                task_type="unclear",
                answer="Please clarify whether you want a database answer or a forecast.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=None,
                latency_ms={"total": round((time.perf_counter() - started) * 1000, 1)},
            )

        if route == "sql":
            response = await self._run_sql(request)
        elif route == "scenario":
            response = await self._run_scenario(request)
        elif route == "anomaly":
            response = await self._run_anomaly(request)
        else:
            response = await self._run_forecast(request)

        response.latency_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("router", self._router_node)
        builder.add_node("sql", self._sql_node)
        builder.add_node("forecast", self._forecast_node)
        builder.add_node("scenario", self._scenario_node)
        builder.add_node("anomaly", self._anomaly_node)
        builder.add_node("clarify", self._clarify_node)
        builder.set_entry_point("router")
        builder.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {"sql": "sql", "forecast": "forecast", "scenario": "scenario", "anomaly": "anomaly", "unclear": "clarify"},
        )
        builder.add_edge("sql", END)
        builder.add_edge("forecast", END)
        builder.add_edge("scenario", END)
        builder.add_edge("anomaly", END)
        builder.add_edge("clarify", END)
        return builder.compile()

    async def _router_node(self, state: dict[str, Any]) -> dict[str, Any]:
        request = QueryRequest.model_validate(state["request"])
        return {
            "request": request.model_dump(),
            "route": await route_question(
                question=request.question,
                mode=request.mode,
                metric_registry=self.services.metric_registry,
                table_names=self.services.allowed_tables,
            )
        }

    async def _sql_node(self, state: dict[str, Any]) -> dict[str, Any]:
        request = QueryRequest.model_validate(state["request"])
        response = await self._run_sql(request)
        return {"response": response.model_dump()}

    async def _forecast_node(self, state: dict[str, Any]) -> dict[str, Any]:
        request = QueryRequest.model_validate(state["request"])
        response = await self._run_forecast(request)
        return {"response": response.model_dump()}

    async def _scenario_node(self, state: dict[str, Any]) -> dict[str, Any]:
        request = QueryRequest.model_validate(state["request"])
        response = await self._run_scenario(request)
        return {"response": response.model_dump()}

    async def _anomaly_node(self, state: dict[str, Any]) -> dict[str, Any]:
        request = QueryRequest.model_validate(state["request"])
        response = await self._run_anomaly(request)
        return {"response": response.model_dump()}

    async def _clarify_node(self, state: dict[str, Any]) -> dict[str, Any]:
        response = QueryResponse(
            status="blocked",
            task_type="unclear",
            answer="Please clarify whether you want a database answer or a forecast.",
            confidence=0.0,
            warnings=_warning_items(
                [warning("data_quality", "The request could not be confidently routed to SQL or forecasting.")]
            ),
            artifacts=None,
            latency_ms={"total": 0.0},
        )
        return {"response": response.model_dump()}

    async def _run_sql(self, request: QueryRequest) -> QueryResponse:
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []

        from infra.vanna_engine import vn
        import pandas as pd
        from orchestrator.nodes.forecast.extender import attempt_extend_forecast

        t0 = time.perf_counter()
        
        try:
            # 1. Ask Vanna to generate SQL
            sql_query = vn.generate_sql(question=request.question)
            latency["vanna_generation"] = round((time.perf_counter() - t0) * 1000, 1)

            # Extract any dates to see if they want a future date
            import re
            date_matches = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', request.question)
            
            # Simple heuristic: If we found a date in the question, check if it's out of bounds
            # For simplicity, we just extract the first table name from the SQL.
            table_match = re.search(r'FROM\s+([a-zA-Z0-9_]+)', sql_query, re.IGNORECASE)
            extended = False
            if table_match and date_matches:
                t_name = table_match.group(1).replace("_forecast", "")
                try:
                    target_dt = pd.to_datetime(date_matches[0])
                    # Phase 4: Dynamic Out-Of-Bounds Extension!
                    extended = attempt_extend_forecast(t_name, target_dt)
                except Exception:
                    pass
                    
            if extended:
                warnings.append(warning("performance", f"Requested date was beyond the auto-forecast horizon. System dynamically extended the forecast via model inference."))
                # Re-generate SQL just in case, though the previous one was probably fine
                sql_query = vn.generate_sql(question=request.question)

            # 2. Execute SQL
            t0 = time.perf_counter()
            df = vn.run_sql(sql_query)
            latency["vanna_execution"] = round((time.perf_counter() - t0) * 1000, 1)
            
            rows = df.to_dict("records")
            row_count = len(rows)

            answer = self._summarize_sql_result(request.question, rows)

            return QueryResponse(
                status="ok" if not warnings else "degraded",
                task_type="sql",
                answer=answer,
                confidence=0.9, # Vanna accuracy
                warnings=_warning_items(warnings),
                artifacts=SQLArtifact(
                    generated_sql=sql_query,
                    selected_tables=[table_match.group(1)] if table_match else [],
                    validation_status="valid",
                    row_count=row_count,
                    preview_rows=rows[:50], # Limit payload
                ),
                latency_ms=latency,
            )

        except Exception as exc:
            warnings.append(warning("model_risk", f"Vanna failed to process request safely: {exc}"))
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="I could not produce or execute a SQL query for that request.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=None,
                latency_ms=latency,
            )


    async def _run_forecast(self, request: QueryRequest) -> QueryResponse:
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []

        t0 = time.perf_counter()
        
        # LLM-driven metric resolution — understands natural language ↔ metric mapping
        metric, llm_filters = await self.services.metric_registry.resolve_with_llm(
            question=request.question,
            metric=request.filters.get("metric") if request.filters else None,
            grain=request.grain
        )

        if not metric:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="No data has been uploaded yet. Please upload a CSV or Excel file first.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=None,
                latency_ms=latency,
            )

    async def _run_forecast(self, request: QueryRequest) -> QueryResponse:
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []

        from infra.vanna_engine import vn
        import pandas as pd
        from orchestrator.nodes.forecast.extender import attempt_extend_forecast
        
        t0 = time.perf_counter()
        
        try:
            # 1. Ask Vanna to generate SQL. Vanna's prompt should pull from _forecast table natively.
            # We add a hint because forecasting needs both history and future.
            hinted_q = f"Select period and value from the relevant _forecast table. {request.question}"
            sql_query = vn.generate_sql(question=hinted_q)
            latency["vanna_generation"] = round((time.perf_counter() - t0) * 1000, 1)
            
            # Extract any dates to see if they want a future date
            import re
            date_matches = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}', request.question)
            table_match = re.search(r'FROM\s+([a-zA-Z0-9_]+forecast)', sql_query, re.IGNORECASE)
            
            extended = False
            if table_match and date_matches:
                t_name = table_match.group(1).replace("_forecast", "")
                try:
                    target_dt = pd.to_datetime(date_matches[0])
                    extended = attempt_extend_forecast(t_name, target_dt)
                except Exception:
                    pass
                    
            if extended:
                warnings.append(warning("performance", f"Requested date was beyond the auto-forecast horizon. Extended automatically."))
                sql_query = vn.generate_sql(question=hinted_q)

            # 2. Execute SQL
            t0 = time.perf_counter()
            df = vn.run_sql(sql_query)
            latency["vanna_execution"] = round((time.perf_counter() - t0) * 1000, 1)
            
            if df.empty or 'period' not in df.columns or 'value' not in df.columns:
                return QueryResponse(
                    status="blocked",
                    task_type="forecast",
                    answer="No valid forecast data found for that request.",
                    confidence=0.0,
                    warnings=_warning_items(warnings),
                    artifacts=None,
                    latency_ms=latency,
                )
                
            # Parse into ForecastArtifact
            df['period'] = pd.to_datetime(df['period'])
            
            if 'is_forecast' in df.columns:
                history_df = df[df['is_forecast'] == False]
                forecast_df = df[df['is_forecast'] == True]
            else:
                # If schema lacks it somehow, assume all history
                history_df = df
                forecast_df = pd.DataFrame(columns=df.columns)
                
            history = [{"period": row['period'].to_pydatetime(), "value": float(row['value'])} for _, row in history_df.iterrows()]
            point_forecast = [{"period": row['period'].to_pydatetime(), "value": float(row['value'])} for _, row in forecast_df.iterrows()]
            
            answer = f"**Forecast Complete.** Found {len(history)} historical data points and {len(point_forecast)} forecasted estimates matching your query."

            full_baseline = [self._serialize_row(item) for item in history]

            return QueryResponse(
                status="ok" if not warnings else "degraded",
                task_type="forecast",
                answer=answer,
                confidence=0.9,
                warnings=_warning_items(warnings),
                artifacts=ForecastArtifact(
                    series_id="vanna_generated",
                    baseline=full_baseline,
                    point_forecast=[self._serialize_row(item) for item in point_forecast],
                    prediction_intervals=[], # Not eagerly calculated to save space
                    anomalies=[], 
                    backtest_metrics={},
                ),
                latency_ms=latency,
            )

        except Exception as exc:
            warnings.append(warning("model_risk", f"Vanna failed to process request safely: {exc}"))
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="I could not produce or execute a SQL forecast query for that request.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=None,
                latency_ms=latency,
            )


    async def _run_scenario(self, request: QueryRequest) -> QueryResponse:
        """Run baseline forecast then apply a scenario multiplier for side-by-side comparison."""
        import re
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []

        # Extract percentage from question dynamically
        match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', request.question)
        factor = 1.0 + (float(match.group(1)) / 100.0) if match else 1.10
        scenario_desc = f"{'+' if factor >= 1 else ''}{(factor - 1) * 100:.0f}% adjustment"

        # Get baseline forecast via the standard pipeline
        t0 = time.perf_counter()
        baseline_response = await self._run_forecast(request)
        latency["baseline_forecast"] = round((time.perf_counter() - t0) * 1000, 1)

        if baseline_response.status == "blocked" or not baseline_response.artifacts:
            return baseline_response

        base_art = baseline_response.artifacts
        baseline_points = base_art.point_forecast
        baseline_intervals = base_art.prediction_intervals

        # Apply scenario multiplier
        scenario_points = [{**p, "value": round(p["value"] * factor, 2)} for p in baseline_points]
        scenario_intervals = [
            {
                **iv,
                "low_80": round(iv["low_80"] * factor, 2),
                "high_80": round(iv["high_80"] * factor, 2),
                "low_95": round(iv.get("low_95", iv["low_80"]) * factor, 2),
                "high_95": round(iv.get("high_95", iv["high_80"]) * factor, 2),
            }
            for iv in baseline_intervals
        ]

        # Build comparison summary
        if baseline_points and scenario_points:
            base_avg = sum(p["value"] for p in baseline_points) / len(baseline_points)
            scen_avg = sum(p["value"] for p in scenario_points) / len(scenario_points)
            diff_pct = ((scen_avg - base_avg) / base_avg * 100) if base_avg else 0
            comparison = f"Under the {scenario_desc} scenario, the average forecast is {scen_avg:,.0f} (vs {base_avg:,.0f} baseline). That's a {diff_pct:+.1f}% change."
        else:
            comparison = "Unable to compute comparison."

        answer = f"**Scenario Analysis ({scenario_desc}):** {comparison}"

        return QueryResponse(
            status="ok",
            task_type="scenario",
            answer=answer,
            confidence=baseline_response.confidence,
            warnings=baseline_response.warnings,
            artifacts=ScenarioArtifact(
                series_id=base_art.series_id,
                baseline_forecast=baseline_points,
                scenario_forecast=scenario_points,
                baseline_intervals=baseline_intervals,
                scenario_intervals=scenario_intervals,
                scenario_description=scenario_desc,
                comparison_summary=comparison,
            ),
            latency_ms=latency,
        )

    async def _run_anomaly(self, request: QueryRequest) -> QueryResponse:
        """Dedicated anomaly detection on the historical data."""
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []

        t0 = time.perf_counter()
        metric, llm_filters = await self.services.metric_registry.resolve_with_llm(
            question=request.question,
            metric=request.filters.get("metric") if request.filters else None,
            grain=request.grain
        )

        if not metric:
            return QueryResponse(
                status="blocked",
                task_type="anomaly",
                answer="No data has been uploaded yet. Please upload a CSV or Excel file first.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "No metrics available for anomaly detection.")]),
                artifacts=None,
                latency_ms=latency,
            )

        try:
            effective_filters = {**(request.filters or {}), **(llm_filters or {})}
            sql = self.services.metric_registry.render_sql(metric, effective_filters)
            with duckdb_connection(read_only=True) as conn:
                df = conn.execute(sql).fetchdf()
            import pandas as pd
            period_col = df.select_dtypes(include=['datetime', 'datetimetz', 'object']).columns[0]
            value_col = df.select_dtypes(include=['number']).columns[0]
            df[period_col] = pd.to_datetime(df[period_col], errors='coerce')
            df = df.dropna(subset=[period_col, value_col])
            history = [{"period": row[period_col].to_pydatetime(), "value": float(row[value_col])} for _, row in df.iterrows()]
        except Exception as e:
            return QueryResponse(
                status="blocked",
                task_type="anomaly",
                answer=f"Failed to extract data for anomaly detection: {e}",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", str(e))]),
                artifacts=None,
                latency_ms=latency,
            )

        latency["extraction"] = round((time.perf_counter() - t0) * 1000, 1)

        grain = request.grain or metric.default_grain
        t0 = time.perf_counter()
        prepared = prepare_series(history, grain)
        series = prepared["series"]
        latency["prep"] = round((time.perf_counter() - t0) * 1000, 1)

        # Enhanced z-score anomaly detection
        if len(series) < 3:
            return QueryResponse(
                status="blocked",
                task_type="anomaly",
                answer="Not enough data points for anomaly detection.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "Need at least 3 data points.")]),
                artifacts=None,
                latency_ms=latency,
            )

        import statistics
        values = [p["value"] for p in series]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 1.0
        threshold = 2.0

        anomalies = []
        for point in series:
            z_score = (point["value"] - mean_val) / std_val if std_val > 0 else 0
            if abs(z_score) > threshold:
                direction = "spike" if z_score > 0 else "dip"
                severity = "high" if abs(z_score) > 3 else "medium"
                anomalies.append({
                    "period": point["period"],
                    "actual": point["value"],
                    "z_score": round(z_score, 2),
                    "mean": round(mean_val, 2),
                    "severity": severity,
                    "direction": direction,
                    "explanation": f"Value of {point['value']:,.0f} is {abs(z_score):.1f} standard deviations {'above' if z_score > 0 else 'below'} the average of {mean_val:,.0f}.",
                })

        latency["anomaly_detection"] = round((time.perf_counter() - t0) * 1000, 1)

        if anomalies:
            answer_parts = [f"**Anomaly Report for {metric.label}:** Found {len(anomalies)} anomalous data point(s).\n"]
            for a in anomalies[:5]:
                period_str = a['period'].strftime('%Y-%m-%d') if hasattr(a['period'], 'strftime') else str(a['period'])
                answer_parts.append(f"- **{period_str}**: {a['explanation']} ({a['severity'].upper()} {a['direction']})")
            if len(anomalies) > 5:
                answer_parts.append(f"\n...and {len(anomalies) - 5} more.")
            answer_parts.append("\n**Suggested next steps:** Investigate the flagged periods for external events, data entry errors, or system issues.")
            answer = "\n".join(answer_parts)
        else:
            answer = f"**Anomaly Report for {metric.label}:** No significant anomalies detected. All data points fall within normal statistical bounds (±2σ)."

        return QueryResponse(
            status="ok",
            task_type="anomaly",
            answer=answer,
            confidence=0.8,
            warnings=_warning_items(warnings),
            artifacts=ForecastArtifact(
                series_id=metric.key,
                baseline=[self._serialize_row(p) for p in series],
                point_forecast=[],
                prediction_intervals=[],
                anomalies=[self._serialize_row(a) for a in anomalies],
                backtest_metrics={"anomaly_count": len(anomalies), "mean": round(mean_val, 2), "std": round(std_val, 2)},
            ),
            latency_ms=latency,
        )

    @staticmethod
    def _serialize_row(item: dict[str, Any]) -> dict[str, Any]:
        return {key: (value.isoformat() if hasattr(value, "isoformat") else value) for key, value in item.items()}

    @staticmethod
    def _summarize_sql_result(question: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "The query ran successfully but returned no rows."
        first_row = rows[0]
        summary = ", ".join(f"{key}={value}" for key, value in list(first_row.items())[:3])
        return f"Top result for '{question}': {summary}"

    @staticmethod
    def _summarize_forecast(metric_label: str, point_forecast: list[dict[str, Any]], intervals: list[dict[str, Any]], anomalies: list[dict[str, Any]], backtest: dict | None = None, confidence: float = 0.0, history_len: int = 0) -> str:
        if not point_forecast:
            return f"No forecast could be generated for {metric_label}."

        n = len(point_forecast)
        first_val = point_forecast[0]["value"]
        last_val = point_forecast[-1]["value"]
        growth = ((last_val - first_val) / first_val * 100) if first_val else 0

        first_interval = intervals[0] if intervals else {}
        last_interval = intervals[-1] if intervals else {}

        parts = [f"**{metric_label.title()} Forecast ({n} periods):**"]
        parts.append(f"Central estimate: {first_val:,.0f} → {last_val:,.0f} ({growth:+.1f}% change).")
        if first_interval:
            parts.append(f"Range: {first_interval.get('low_80', first_val):,.0f} – {last_interval.get('high_80', last_val):,.0f}.")

        # Transparency section
        transparency = []
        if history_len:
            transparency.append(f"{history_len} historical data points used")
        if confidence:
            transparency.append(f"confidence: {confidence:.0%}")
        if backtest:
            if backtest.get("beats_baseline"):
                transparency.append("model beats seasonal baseline ✅")
            else:
                transparency.append("model did not beat baseline ⚠️")
            if backtest.get("coverage_80") is not None:
                transparency.append(f"80% interval coverage: {backtest['coverage_80']:.0%}")
        if transparency:
            parts.append(f"\n📊 *Transparency: {', '.join(transparency)}.*")

        if anomalies:
            parts.append(f"\n🚨 **Warning:** {len(anomalies)} anomalous point(s) detected in historical data.")

        return "\n".join(parts)
