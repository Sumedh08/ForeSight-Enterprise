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
from dataclasses import replace
from datetime import datetime
from typing import Any

from api.models.schemas import ForecastArtifact, QueryRequest, QueryResponse, SQLArtifact, ScenarioArtifact, TrainingArtifact, WarningItem
from components.connectors import DatabaseSchema, serialize_schema
from components.forecasting import SeasonalNaiveForecaster
from infra.metrics import warning
from infra.nim_gateway import nim_gateway
from infra.runtime_services import RuntimeServices
from infra.settings import settings
from infra.training_store import TrainingJobRecord
from infra.vanna_engine import VannaSemanticError, vanna_engine
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

        try:
            sql, tables = await self._resolve_grounding_sql(
                question=request.question,
                schema=schema,
                connector=connector,
                allowed_tables=allowed_tables,
                cache_key=f"sql::{request.question}",
                row_cap=settings.default_row_cap,
                warnings=warnings,
                series_mode=False,
                latency=latency,
            )
        except VannaSemanticError:
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="I couldn't map that request to a known metric. Define a new metric and try again.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "Vanna could not map the request to a known metric.")]),
                artifacts=None,
                latency_ms=latency,
            )
        except ValueError as exc:
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="The generated SQL failed safety validation and was blocked.",
                confidence=0.0,
                warnings=_warning_items([warning("safety_block", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )
        except Exception as exc:
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="Autonomous SQL generation is unavailable right now.",
                confidence=0.0,
                warnings=_warning_items(warnings or [warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )

        t0 = time.perf_counter()
        try:
            execution = connector.execute(sql, preview_limit=50)
            rows = execution.rows
        except Exception as exc:
            if latency.get("cache_hit", 0.0) == 1.0:
                self.services.vanna_cache.invalidate(question=f"sql::{request.question}", sql=sql)
            warnings.append(warning("degraded_mode", f"Execution failed: {exc}"))
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="SQL execution failed on the active database.",
                confidence=0.0,
                warnings=_warning_items(warnings),
                artifacts=SQLArtifact(
                    generated_sql=sql,
                    selected_tables=tables,
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
                generated_sql=sql,
                selected_tables=tables,
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

        latency["schema_introspection"] = round((time.perf_counter() - t0) * 1000, 1)
        allowed_tables = {table.name for table in schema.tables}

        try:
            sql, tables = await self._resolve_grounding_sql(
                question=request.question,
                schema=schema,
                connector=connector,
                allowed_tables=allowed_tables,
                cache_key=f"series::{request.question}",
                row_cap=5000,
                warnings=warnings,
                series_mode=True,
                latency=latency,
            )
        except VannaSemanticError:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="I couldn't map that request to a known metric. Define a new metric and try again.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "Vanna could not map the forecast request to a known metric.")]),
                artifacts=None,
                latency_ms=latency,
            )
        except ValueError as exc:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="The grounded forecast query failed safety validation and was blocked.",
                confidence=0.0,
                warnings=_warning_items([warning("safety_block", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )
        except Exception as exc:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="Vanna could not ground a historical time series for this request.",
                confidence=0.0,
                warnings=_warning_items(warnings or [warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )

        t0 = time.perf_counter()
        try:
            execution = connector.execute(sql, preview_limit=5000)
            grounded_rows = execution.rows
        except Exception as exc:
            if latency.get("cache_hit", 0.0) == 1.0:
                self.services.vanna_cache.invalidate(question=f"series::{request.question}", sql=sql)
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="The grounded forecast query failed to execute.",
                confidence=0.0,
                warnings=_warning_items([warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )
        latency["grounding_execution"] = round((time.perf_counter() - t0) * 1000, 1)

        historical_series = self._rows_to_series(grounded_rows)
        if len(historical_series) < 8:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="The system could not ground enough historical data for forecasting. Try another query or provide more data.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "Forecast grounding requires a historical `period`/`value` series.")]),
                artifacts=None,
                latency_ms=latency,
            )

        job = await self._resolve_training_job(request=request, table_names=tables)
        if job is None:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="I could not find a trained forecasting metric for this request. Define a new metric and try again.",
                confidence=0.0,
                warnings=_warning_items([warning("data_quality", "No matching MindsDB training job was found for the grounded metric.")]),
                artifacts=None,
                latency_ms=latency,
            )

        t0 = time.perf_counter()
        refreshed_job = await self._refresh_training_job(job)
        latency["mindsdb_status_lookup"] = round((time.perf_counter() - t0) * 1000, 1)
        preview = self._build_preview_baseline(
            historical_series=historical_series,
            question=request.question,
            requested_horizon=request.horizon,
            requested_grain=request.grain,
        )

        if refreshed_job.state in {"queued", "training"}:
            return QueryResponse(
                status="training",
                task_type="forecast",
                answer=f"AI Learning: {refreshed_job.progress_pct}% Ready. Returning a seasonal naive preview while MindsDB finishes training `{refreshed_job.predictor_name}`.",
                confidence=0.56,
                warnings=_warning_items(warnings),
                artifacts=TrainingArtifact(
                    series_id=refreshed_job.series_id,
                    predictor_name=refreshed_job.predictor_name,
                    state=refreshed_job.state,
                    progress_pct=refreshed_job.progress_pct,
                    message=refreshed_job.message,
                    poll_after_ms=refreshed_job.poll_after_ms,
                    preview_baseline=preview,
                ),
                latency_ms=latency,
            )

        if refreshed_job.state == "failed":
            if preview:
                return QueryResponse(
                    status="degraded",
                    task_type="forecast",
                    answer=(
                        f"MindsDB predictor `{refreshed_job.predictor_name}` is unavailable, so I am returning "
                        "a deterministic seasonal naive baseline instead."
                    ),
                    confidence=0.42,
                    warnings=_warning_items(warnings + [warning("model_risk", refreshed_job.message)]),
                    artifacts=ForecastArtifact(
                        series_id=refreshed_job.series_id,
                        baseline=self._serialize_history(historical_series, limit=24),
                        point_forecast=preview,
                        prediction_intervals=[],
                        anomalies=[],
                        backtest_metrics={
                            "source": "seasonal_naive_fallback",
                            "predictor_failure": refreshed_job.message,
                            "grounded_tables": tables,
                        },
                    ),
                    latency_ms=latency,
                )
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer=f"MindsDB could not build predictor `{refreshed_job.predictor_name}`.",
                confidence=0.0,
                warnings=_warning_items(warnings + [warning("model_risk", refreshed_job.message)]),
                artifacts=TrainingArtifact(
                    series_id=refreshed_job.series_id,
                    predictor_name=refreshed_job.predictor_name,
                    state="failed",
                    progress_pct=refreshed_job.progress_pct,
                    message=refreshed_job.message,
                    poll_after_ms=refreshed_job.poll_after_ms,
                    preview_baseline=preview,
                ),
                latency_ms=latency,
            )

        t0 = time.perf_counter()
        try:
            rows = await self.services.mindsdb_client.run_predictor(
                predictor=refreshed_job.predictor_name,
                row_cap=max(100, request.horizon or 26),
            )
            point_forecast = self._rows_to_forecast_points(
                rows,
                period_hint=refreshed_job.date_column,
                value_hint=refreshed_job.value_column,
            )
        except Exception as exc:
            state, progress_pct, message = await self.services.mindsdb_client.get_predictor_state(refreshed_job.predictor_name)
            try:
                updated = self.services.training_store.update_job(
                    refreshed_job.series_id,
                    state=state,
                    progress_pct=progress_pct,
                    message=message,
                )
            except Exception:
                updated = None
            current_job = updated or replace(refreshed_job, state=state, progress_pct=progress_pct, message=message)
            if state in {"queued", "training"}:
                return QueryResponse(
                    status="training",
                    task_type="forecast",
                    answer=f"AI Learning: {current_job.progress_pct}% Ready. MindsDB is still training `{current_job.predictor_name}`.",
                    confidence=0.56,
                    warnings=_warning_items(warnings),
                    artifacts=TrainingArtifact(
                        series_id=current_job.series_id,
                        predictor_name=current_job.predictor_name,
                        state=current_job.state,
                        progress_pct=current_job.progress_pct,
                        message=current_job.message,
                        poll_after_ms=current_job.poll_after_ms,
                        preview_baseline=preview,
                    ),
                    latency_ms=latency,
                )
            if preview:
                return QueryResponse(
                    status="degraded",
                    task_type="forecast",
                    answer=(
                        f"MindsDB predictor `{current_job.predictor_name}` is unavailable, so I am returning "
                        "a deterministic seasonal naive baseline instead."
                    ),
                    confidence=0.42,
                    warnings=_warning_items(warnings + [warning("model_risk", current_job.message)]),
                    artifacts=ForecastArtifact(
                        series_id=current_job.series_id,
                        baseline=self._serialize_history(historical_series, limit=24),
                        point_forecast=preview,
                        prediction_intervals=[],
                        anomalies=[],
                        backtest_metrics={
                            "source": "seasonal_naive_fallback",
                            "predictor_failure": current_job.message,
                            "grounded_tables": tables,
                        },
                    ),
                    latency_ms=latency,
                )
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="MindsDB inference failed for this predictor.",
                confidence=0.0,
                warnings=_warning_items(warnings + [warning("degraded_mode", str(exc))]),
                artifacts=None,
                latency_ms=latency,
            )
        latency["mindsdb_inference"] = round((time.perf_counter() - t0) * 1000, 1)

        if not point_forecast:
            return QueryResponse(
                status="blocked",
                task_type="forecast",
                answer="MindsDB returned no forecast points for this predictor.",
                confidence=0.0,
                warnings=_warning_items(warnings + [warning("model_risk", "Predictor output did not expose a usable `period`/`value` series.")]),
                artifacts=None,
                latency_ms=latency,
            )

        answer = await self._summarize_forecast(
            question=request.question,
            historical_series=historical_series,
            forecast_points=point_forecast,
            predictor_name=refreshed_job.predictor_name,
        )
        return QueryResponse(
            status="ok",
            task_type="forecast",
            answer=answer,
            confidence=0.83,
            warnings=_warning_items(warnings),
            artifacts=ForecastArtifact(
                series_id=refreshed_job.series_id,
                baseline=self._serialize_history(historical_series, limit=24),
                point_forecast=point_forecast,
                prediction_intervals=[],
                anomalies=[],
                backtest_metrics={
                    "source": "mindsdb",
                    "grounded_tables": tables,
                    "cache_hit": bool(latency.get("cache_hit", 0.0)),
                },
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

    async def _resolve_grounding_sql(
        self,
        *,
        question: str,
        schema: DatabaseSchema,
        connector: Any,
        allowed_tables: set[str],
        cache_key: str,
        row_cap: int,
        warnings: list[dict[str, str]],
        series_mode: bool,
        latency: dict[str, float],
    ) -> tuple[str, list[str]]:
        t0 = time.perf_counter()
        cache_entry = self.services.vanna_cache.lookup(cache_key)
        latency["semantic_cache_lookup"] = round((time.perf_counter() - t0) * 1000, 1)
        latency["cache_hit"] = 0.0

        if cache_entry is not None:
            try:
                guarded = guard_sql(
                    cache_entry.sql,
                    dialect=schema.dialect,
                    allowed_tables=allowed_tables,
                    row_cap=row_cap,
                )
                connector.dry_run(guarded.sql)
                latency["cache_hit"] = 1.0
                latency["sql_validation"] = 0.0
                return guarded.sql, guarded.tables
            except Exception:
                self.services.vanna_cache.invalidate(question=cache_key, sql=cache_entry.sql)
                warnings.append(warning("performance", "Semantic cache entry was stale and has been invalidated."))

        schema_context = serialize_schema(schema)
        t0 = time.perf_counter()
        
        # When grounding a forecast, we need the historical DATA for the metric, 
        # not a query filtered for the future date in the question.
        vanna_question = question
        if series_mode:
            vanna_question = f"Select the full historical time-series (all dates and values) needed to answer: {question}"
            
        generated_sql = await vanna_engine.generate_sql(vanna_question)
        latency["vanna_generation"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        guarded = guard_sql(
            generated_sql,
            dialect=schema.dialect,
            allowed_tables=allowed_tables,
            row_cap=row_cap,
        )
        connector.dry_run(guarded.sql)
        latency["sql_validation"] = round((time.perf_counter() - t0) * 1000, 1)
        self.services.vanna_cache.remember(question=cache_key, sql=guarded.sql, selected_tables=guarded.tables)
        return guarded.sql, guarded.tables

    async def _resolve_training_job(self, *, request: QueryRequest, table_names: list[str]) -> TrainingJobRecord | None:
        if request.series_id:
            direct = self.services.training_store.get_job(series_id=request.series_id)
            if direct is not None:
                return direct
        return self.services.training_store.match_job(
            question=request.question,
            metric_hint=request.metric,
            table_names=table_names,
        )

    async def _refresh_training_job(self, job: TrainingJobRecord) -> TrainingJobRecord:
        if job.state not in {"queued", "training"}:
            return job
        try:
            state, progress_pct, message = await self.services.mindsdb_client.get_predictor_state(job.predictor_name)
            try:
                updated = self.services.training_store.update_job(
                    job.series_id,
                    state=state,
                    progress_pct=progress_pct,
                    message=message,
                )
            except Exception:
                updated = None
            return updated or replace(job, state=state, progress_pct=progress_pct, message=message)
        except Exception:
            return job

    async def _summarize_forecast(
        self,
        *,
        question: str,
        historical_series: list[dict[str, Any]],
        forecast_points: list[dict[str, Any]],
        predictor_name: str,
    ) -> str:
        if nim_gateway.enabled:
            try:
                prompt = (
                    "You are an enterprise analytics analyst.\n"
                    "Write a short narrative using only the historical series and forecast output below.\n"
                    "Do not mention SQL, models, or hidden steps. Mention magnitude and direction clearly.\n\n"
                    f"Question: {question}\n"
                    f"Predictor: {predictor_name}\n"
                    f"Historical series: {json.dumps(self._serialize_history(historical_series, limit=12), default=str)}\n"
                    f"Forecast points: {json.dumps(forecast_points[:12], default=str)}"
                )
                response = await nim_gateway.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=220)
                if response.strip():
                    return response.strip()
            except Exception:
                pass

        first = forecast_points[0]
        last = forecast_points[-1]
        return (
            f"Forecast complete using predictor `{predictor_name}`. "
            f"The outlook moves from {round(float(first['value']), 4)} to {round(float(last['value']), 4)} "
            f"across {len(forecast_points)} projected period(s)."
        )

    def _build_preview_baseline(
        self,
        *,
        historical_series: list[dict[str, Any]],
        question: str,
        requested_horizon: int | None,
        requested_grain: str | None,
    ) -> list[dict[str, Any]]:
        if len(historical_series) < 3:
            return []
        grain = self._normalize_grain(requested_grain) or self._infer_grain(historical_series)
        horizon = self._resolve_forecast_horizon(
            question=question,
            grain=grain,
            last_period=historical_series[-1]["period"],
            requested_horizon=requested_horizon,
        )
        model = SeasonalNaiveForecaster()
        points = model.predict(historical_series, horizon=horizon, grain=grain)
        return [
            {"period": point.period.isoformat(), "value": round(float(point.value), 6)}
            for point in points
        ]

    @staticmethod
    def _serialize_history(series: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        return [
            {"period": item["period"].isoformat(), "value": round(float(item["value"]), 6)}
            for item in series[-min(len(series), limit):]
        ]

    def _rows_to_series(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        points = self._rows_to_forecast_points(rows)
        series: list[dict[str, Any]] = []
        for point in points:
            period = self._parse_datetime(point.get("period"))
            if period is None:
                continue
            series.append({"period": period, "value": float(point["value"])})
        return series

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

    def _rows_to_forecast_points(
        self,
        rows: list[dict[str, Any]],
        *,
        period_hint: str | None = None,
        value_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        columns = list(rows[0].keys())
        period_key = self._detect_period_column(columns, rows, preferred=period_hint)
        value_key = self._detect_numeric_column(columns, rows, preferred=value_hint)
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
    def _normalize_grain(grain: str | None) -> str | None:
        if grain is None:
            return None
        lowered = str(grain).strip().lower()
        mapping = {
            "d": "day",
            "day": "day",
            "daily": "day",
            "w": "week",
            "week": "week",
            "weekly": "week",
            "m": "month",
            "month": "month",
            "monthly": "month",
        }
        return mapping.get(lowered)

    def _infer_grain(self, series: list[dict[str, Any]]) -> str:
        if len(series) < 3:
            return "week"
        day_diffs: list[float] = []
        for index in range(1, len(series)):
            previous = series[index - 1]["period"]
            current = series[index]["period"]
            delta_days = (current - previous).total_seconds() / 86400
            if delta_days > 0:
                day_diffs.append(delta_days)
        if not day_diffs:
            return "week"
        median_days = statistics.median(day_diffs)
        if median_days <= 2:
            return "day"
        if median_days <= 12:
            return "week"
        return "month"

    def _resolve_forecast_horizon(
        self,
        *,
        question: str,
        grain: str,
        last_period: datetime,
        requested_horizon: int | None,
    ) -> int:
        if isinstance(requested_horizon, int) and requested_horizon > 0:
            return min(requested_horizon, 520)

        target_date = self._extract_target_date(question)
        if target_date is not None and target_date > last_period:
            delta_days = (target_date - last_period).total_seconds() / 86400
            if grain == "day":
                return min(max(1, int(delta_days) + 1), 520)
            if grain == "week":
                return min(max(1, int(delta_days / 7) + 1), 520)
            month_span = (target_date.year - last_period.year) * 12 + (target_date.month - last_period.month)
            if target_date.day > last_period.day:
                month_span += 1
            return min(max(1, month_span), 520)

        defaults = {"day": 30, "week": 12, "month": 6}
        return defaults.get(grain, 12)

    @staticmethod
    def _extract_target_date(question: str) -> datetime | None:
        patterns = [
            (r"\b\d{4}-\d{2}-\d{2}\b", ["%Y-%m-%d"]),
            (r"\b\d{1,2}/\d{1,2}/\d{4}\b", ["%d/%m/%Y", "%m/%d/%Y"]),
            (r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", ["%d %b %Y", "%d %B %Y"]),
        ]
        for regex, formatters in patterns:
            match = re.search(regex, question)
            if not match:
                continue
            token = match.group(0)
            for formatter in formatters:
                try:
                    return datetime.strptime(token, formatter)
                except ValueError:
                    continue
        return None

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
    def _detect_period_column(columns: list[str], rows: list[dict[str, Any]], preferred: str | None = None) -> str | None:
        if preferred:
            for column in columns:
                if column.lower() == preferred.lower():
                    return column
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
    def _detect_numeric_column(columns: list[str], rows: list[dict[str, Any]], preferred: str | None = None) -> str | None:
        blocked = {"is_forecast", "confidence", "lower", "upper"}
        if preferred:
            for column in columns:
                if column.lower() == preferred.lower():
                    try:
                        float(rows[0].get(column))
                        return column
                    except Exception:
                        pass
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
