"""
Enterprise Analytics Orchestrator
Single runtime path: intent routing -> SQL generation -> SQLGlot AST guard -> read-only execution.
Forecasting uses MindsDB predictors when available.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime
from typing import Any

from api.models.schemas import ForecastArtifact, QueryRequest, QueryResponse, SQLArtifact, ScenarioArtifact, WarningItem
from infra.metrics import warning
from infra.nim_gateway import nim_gateway
from infra.runtime_services import RuntimeServices
from infra.settings import settings
from safety.sql_guard import guard_sql


def _warning_items(items: list[dict[str, str]]) -> list[WarningItem]:
    return [WarningItem(kind=item["kind"], message=item["message"]) for item in items]


def _is_forecast_question(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("forecast", "predict", "future", "next week", "next month"))


def _is_scenario_question(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("what if", "scenario", "increase by", "decrease by", "%"))


def _is_anomaly_question(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ("anomaly", "outlier", "spike", "dip", "unusual"))


def _extract_percent_factor(question: str) -> float:
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", question)
    if not match:
        return 1.10
    return 1.0 + (float(match.group(1)) / 100.0)


def _to_iso(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _schema_context(schema: Any, table_limit: int = 20) -> str:
    lines = [f"Dialect: {getattr(schema, 'dialect', 'unknown')}"]
    for table in schema.tables[:table_limit]:
        lines.append(f"Table {table.name}:")
        for column in table.columns[:30]:
            lines.append(f"- {column.name} ({column.data_type})")
    return "\n".join(lines)


class EnterpriseOrchestrator:
    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    async def run_query(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        route = await self._route_intent(request)
        if route == "sql":
            response = await self._run_sql(request)
        elif route == "forecast":
            response = await self._run_forecast(request)
        elif route == "scenario":
            response = await self._run_scenario(request)
        elif route == "anomaly":
            response = await self._run_anomaly(request)
        else:
            response = QueryResponse(
                status="blocked",
                task_type="unclear",
                answer="Please clarify whether you want SQL analysis, a forecast, anomaly detection, or scenario simulation.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "Intent could not be routed with confidence.")]),
                artifacts=None,
                latency_ms={},
            )

        response.latency_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    async def _route_intent(self, request: QueryRequest) -> str:
        if request.mode != "auto":
            return request.mode

        if nim_gateway.enabled:
            prompt = (
                "Classify intent into one of: sql, forecast, scenario, anomaly, unclear.\n"
                f"Question: {request.question}\n"
                "Reply with only the label."
            )
            try:
                response = await nim_gateway.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=8)
                label = response.strip().lower().strip("'\"")
                if label in {"sql", "forecast", "scenario", "anomaly", "unclear"}:
                    return label
            except Exception:
                pass

        question = request.question
        if _is_scenario_question(question):
            return "scenario"
        if _is_anomaly_question(question):
            return "anomaly"
        if _is_forecast_question(question):
            return "forecast"
        return "sql"

    async def _run_sql(self, request: QueryRequest) -> QueryResponse:
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []
        t0 = time.perf_counter()
        manager = self.services.connection_manager

        try:
            profile = manager.get_active_profile()
            connector = manager.build_connector(profile, read_only=True)
            schema = connector.introspect_schema(sample_limit=2)
            allowed_tables = {table.name for table in schema.tables}
        except Exception as exc:
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="No active database connection is ready. Configure and test a connection first.",
                confidence=0.0,
                warnings=_warning_items([warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )
        latency["schema_introspection"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        try:
            generated_sql = await self.services.wren_client.generate_sql(
                request.question,
                schema_context=_schema_context(schema),
            )
        except Exception as exc:
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="Wren SQL generation is unavailable right now.",
                confidence=0.0,
                warnings=_warning_items([warning("degraded_mode", f"Wren error: {exc}")]),
                artifacts=None,
                latency_ms=latency,
            )
        latency["wren_generation"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        try:
            guarded = guard_sql(
                generated_sql,
                dialect=schema.dialect,
                allowed_tables=allowed_tables,
                row_cap=settings.default_row_cap,
            )
            connector.dry_run(guarded.sql)
        except Exception as exc:
            warnings.append(warning("safety_block", f"SQL blocked by safety guard: {exc}"))
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="The generated SQL failed safety validation and was blocked.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=SQLArtifact(
                    generated_sql=generated_sql,
                    selected_tables=[],
                    validation_status="failed",
                    row_count=0,
                    preview_rows=[],
                ),
                latency_ms=latency,
            )
        latency["sql_validation"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        try:
            execution = connector.execute(guarded.sql, preview_limit=50)
            rows = execution.rows
        except Exception as exc:
            warnings.append(warning("degraded_mode", f"Execution failed: {exc}"))
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="SQL execution failed on the active database.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=SQLArtifact(
                    generated_sql=guarded.sql,
                    selected_tables=guarded.tables,
                    validation_status="valid",
                    row_count=0,
                    preview_rows=[],
                ),
                latency_ms=latency,
            )
        latency["sql_execution"] = round((time.perf_counter() - t0) * 1000, 1)

        answer = await self._summarize_rows(question=request.question, rows=rows)
        status = "ok" if not warnings else "degraded"
        confidence = 0.88 if execution.row_count > 0 else 0.74

        return QueryResponse(
            status=status,
            task_type="sql",
            answer=answer,
            confidence=confidence,
            warnings=_warning_items(warnings),
            artifacts=SQLArtifact(
                generated_sql=guarded.sql,
                selected_tables=guarded.tables,
                validation_status="valid",
                row_count=execution.row_count,
                preview_rows=[{key: _to_iso(value) for key, value in row.items()} for row in rows],
            ),
            latency_ms=latency,
        )

    async def _run_forecast(self, request: QueryRequest) -> QueryResponse:
        latency: dict[str, float] = {}
        warnings: list[dict[str, str]] = []
        t0 = time.perf_counter()
        manager = self.services.connection_manager

        try:
            profile = manager.get_active_profile()
            connector = manager.build_connector(profile, read_only=True)
            schema = connector.introspect_schema(sample_limit=1)
            table_names = [table.name for table in schema.tables]
        except Exception as exc:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="No active database connection is ready for forecasting.",
                confidence=0.0,
                warnings=_warning_items([warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )

        if profile.get("type") in {"postgres", "mysql"}:
            try:
                await self.services.mindsdb_client.ensure_datasource(profile)
            except Exception as exc:
                warnings.append(warning("degraded_mode", f"MindsDB datasource sync warning: {exc}"))

        try:
            predictor = request.series_id or await self.services.mindsdb_client.resolve_predictor(
                question=request.question,
                table_names=table_names,
            )
        except Exception as exc:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="MindsDB predictor discovery failed.",
                confidence=0.0,
                warnings=_warning_items([warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )

        if not predictor:
            warnings.append(warning("data_quality", "No predictor matched this request."))
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="No trained MindsDB predictor is available yet. Upload data and trigger training first.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=None,
                latency_ms=latency,
            )

        try:
            rows = await self.services.mindsdb_client.run_predictor(predictor=predictor, row_cap=200)
        except Exception as exc:
            warnings.append(warning("degraded_mode", str(exc)))
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="MindsDB inference failed.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=None,
                latency_ms=latency,
            )
        latency["mindsdb_inference"] = round((time.perf_counter() - t0) * 1000, 1)

        point_forecast = self._rows_to_forecast_points(rows)
        if not point_forecast:
            warnings.append(warning("model_risk", "Predictor returned rows, but period/value columns could not be resolved."))

        return QueryResponse(
            status="ok" if point_forecast else "degraded",
            task_type="forecast",
            answer=f"Generated forecast using predictor `{predictor}` with {len(point_forecast)} projected points.",
            confidence=0.78 if point_forecast else 0.45,
            warnings=_warning_items(warnings),
            artifacts=ForecastArtifact(
                series_id=predictor,
                baseline=[],
                point_forecast=point_forecast,
                prediction_intervals=[],
                anomalies=[],
                backtest_metrics={},
            ),
            latency_ms=latency,
        )

    async def _run_scenario(self, request: QueryRequest) -> QueryResponse:
        baseline = await self._run_forecast(request)
        if baseline.task_type != "forecast" or baseline.artifacts is None:
            return QueryResponse(
                status="blocked",
                task_type="scenario",
                answer="Scenario simulation requires a successful baseline forecast first.",
                confidence=0.0,
                warnings=baseline.warnings,
                artifacts=None,
                latency_ms=baseline.latency_ms,
            )

        if not isinstance(baseline.artifacts, ForecastArtifact):
            return QueryResponse(
                status="blocked",
                task_type="scenario",
                answer="Scenario simulation requires forecast artifacts.",
                confidence=0.0,
                warnings=baseline.warnings,
                artifacts=None,
                latency_ms=baseline.latency_ms,
            )

        factor = _extract_percent_factor(request.question)
        scenario_points = []
        for row in baseline.artifacts.point_forecast:
            scenario_points.append(
                {
                    "period": row["period"],
                    "value": round(float(row["value"]) * factor, 4),
                }
            )
        description = f"{(factor - 1) * 100:+.1f}% adjustment"
        answer = (
            f"Scenario run complete ({description}). "
            f"Compared {len(scenario_points)} forecast points against the baseline predictor output."
        )

        return QueryResponse(
            status="ok",
            task_type="scenario",
            answer=answer,
            confidence=max(0.0, baseline.confidence - 0.05),
            warnings=baseline.warnings,
            artifacts=ScenarioArtifact(
                series_id=baseline.artifacts.series_id,
                baseline_forecast=baseline.artifacts.point_forecast,
                scenario_forecast=scenario_points,
                baseline_intervals=baseline.artifacts.prediction_intervals,
                scenario_intervals=[],
                scenario_description=description,
                comparison_summary=answer,
            ),
            latency_ms=baseline.latency_ms,
        )

    async def _run_anomaly(self, request: QueryRequest) -> QueryResponse:
        sql_result = await self._run_sql(QueryRequest(question=request.question, mode="sql"))
        if sql_result.status == "blocked" or sql_result.artifacts is None:
            return QueryResponse(
                status="blocked",
                task_type="anomaly",
                answer="Anomaly detection requires a successful SQL extraction of time-series rows.",
                confidence=0.0,
                warnings=sql_result.warnings,
                artifacts=None,
                latency_ms=sql_result.latency_ms,
            )

        if not isinstance(sql_result.artifacts, SQLArtifact):
            return QueryResponse(
                status="blocked",
                task_type="anomaly",
                answer="Anomaly detection could not parse SQL artifact payload.",
                confidence=0.0,
                warnings=sql_result.warnings,
                artifacts=None,
                latency_ms=sql_result.latency_ms,
            )

        rows = sql_result.artifacts.preview_rows
        anomalies, baseline = self._detect_anomalies(rows)
        if not anomalies and not baseline:
            return QueryResponse(
                status="blocked",
                task_type="anomaly",
                answer="No valid period/value columns were available for anomaly detection.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "Anomaly detection expects a timestamp/date and numeric value column.")]),
                artifacts=None,
                latency_ms=sql_result.latency_ms,
            )

        answer = (
            f"Anomaly analysis complete: found {len(anomalies)} anomaly points "
            f"across {len(baseline)} inspected records."
        )
        return QueryResponse(
            status="ok" if anomalies else "degraded",
            task_type="anomaly",
            answer=answer,
            confidence=0.72 if anomalies else 0.58,
            warnings=sql_result.warnings,
            artifacts=ForecastArtifact(
                series_id="anomaly_scan",
                baseline=baseline,
                point_forecast=[],
                prediction_intervals=[],
                anomalies=anomalies,
                backtest_metrics={"records_scanned": len(baseline)},
            ),
            latency_ms=sql_result.latency_ms,
        )

    async def _summarize_rows(self, *, question: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "The query executed successfully but returned no rows."

        if nim_gateway.enabled:
            try:
                prompt = (
                    "Answer the user's question using only the SQL rows below.\n"
                    "Be concise (max 3 sentences), no hallucination.\n\n"
                    f"Question: {question}\n"
                    f"Rows: {json.dumps(rows[:20], default=str)}"
                )
                response = await nim_gateway.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=220)
                if response.strip():
                    return response.strip()
            except Exception:
                pass

        first = rows[0]
        preview = ", ".join(f"{key}={value}" for key, value in list(first.items())[:4])
        return f"Query returned {len(rows)} preview row(s). First row: {preview}."

    def _rows_to_forecast_points(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        columns = list(rows[0].keys())
        period_key = self._detect_period_column(columns, rows)
        value_key = self._detect_numeric_column(columns, rows)
        if not period_key or not value_key:
            return []

        points: list[dict[str, Any]] = []
        for row in rows:
            period = row.get(period_key)
            value = row.get(value_key)
            if value is None:
                continue
            parsed_period = self._parse_datetime(period)
            if parsed_period is None:
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            points.append({"period": parsed_period.isoformat(), "value": numeric})
        return points

    def _detect_anomalies(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not rows:
            return [], []
        columns = list(rows[0].keys())
        period_key = self._detect_period_column(columns, rows)
        value_key = self._detect_numeric_column(columns, rows)
        if not period_key or not value_key:
            return [], []

        series = []
        for row in rows:
            period = self._parse_datetime(row.get(period_key))
            try:
                value = float(row.get(value_key))
            except Exception:
                continue
            if period is None:
                continue
            series.append({"period": period.isoformat(), "value": value})
        if len(series) < 3:
            return [], series

        values = [point["value"] for point in series]
        mean_value = statistics.mean(values)
        stdev = statistics.pstdev(values) or 1.0
        anomalies = []
        for point in series:
            z_score = (point["value"] - mean_value) / stdev
            if abs(z_score) >= 2.0:
                anomalies.append(
                    {
                        "period": point["period"],
                        "actual": point["value"],
                        "z_score": round(z_score, 3),
                        "severity": "high" if abs(z_score) >= 3.0 else "medium",
                        "direction": "spike" if z_score > 0 else "dip",
                    }
                )
        return anomalies, series

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        for pattern in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, pattern)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _detect_period_column(columns: list[str], rows: list[dict[str, Any]]) -> str | None:
        lowered = {column: column.lower() for column in columns}
        for column, lowered_name in lowered.items():
            if any(token in lowered_name for token in ("date", "time", "period", "timestamp", "ds")):
                return column
        for column in columns:
            sample = rows[0].get(column)
            if sample is None:
                continue
            if EnterpriseOrchestrator._parse_datetime(sample) is not None:
                return column
        return None

    @staticmethod
    def _detect_numeric_column(columns: list[str], rows: list[dict[str, Any]]) -> str | None:
        blocked = {"is_forecast", "confidence", "lower", "upper"}
        for column in columns:
            if column.lower() in blocked:
                continue
            sample = rows[0].get(column)
            if sample is None:
                continue
            try:
                float(sample)
                return column
            except Exception:
                continue
        return None
