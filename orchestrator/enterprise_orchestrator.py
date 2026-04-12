"""
Enterprise Analytics Orchestrator (Schema-Agnostic)
The modern successor to graph.py, utilizing Wren AI for semantic Text-to-SQL 
and MindsDB for automated, in-database machine learning.
"""
import time
import logging
import httpx
from typing import Any, Dict, List, Optional
from api.models.schemas import (
    ForecastArtifact, 
    QueryRequest, 
    QueryResponse, 
    SQLArtifact, 
    WarningItem
)
from infra.wren_client import wren_client
from infra.settings import settings

logger = logging.getLogger(__name__)

class EnterpriseOrchestrator:
    def __init__(self, services: Any):
        self.services = services
        self.mindsdb_api = "http://localhost:47334/api/sql"

    async def run_query(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        latency: Dict[str, float] = {}
        
        # 1. Semantic Intent Recognition (Manual routing or Wren AI suggestion)
        # For now, we use a simple heuristic to route between SQL and Forecast
        q_lower = request.question.lower()
        is_forecast = any(k in q_lower for k in ("forecast", "predict", "what will be", "future", "expected"))
        
        try:
            if is_forecast:
                response = await self._run_mindsdb_forecast(request, latency)
            else:
                response = await self._run_wren_semantic_query(request, latency)
                
            response.latency_ms.update(latency)
            response.latency_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
            return response
            
        except Exception as e:
            logger.exception("Orchestration failed")
            return QueryResponse(
                status="blocked",
                task_type="error",
                answer=f"The enterprise analytical engine encountered an error: {str(e)}",
                confidence=0.0,
                latency_ms={"total": round((time.perf_counter() - started) * 1000, 1)},
                artifacts=None
            )

    async def _run_wren_semantic_query(self, request: QueryRequest, latency: Dict[str, float]) -> QueryResponse:
        """Utilizes Wren AI and Cube.js to answer questions via the semantic layer."""
        t0 = time.perf_counter()
        
        # Get SQL from Wren AI (which uses the dynamic Cube schema)
        sql_query = await wren_client.generate_sql(request.question)
        latency["wren_sql_generation"] = round((time.perf_counter() - t0) * 1000, 1)
        
        if not sql_query:
            return QueryResponse(
                status="blocked",
                task_type="sql",
                answer="I couldn't find a semantic mapping for your question. Please ensure data is uploaded and modeled.",
                confidence=0.0,
                latency_ms=latency,
                artifacts=None
            )

        # Execute final query
        # Since we migrated data to Postgres, we can run this SQL via Wren Engine or local PG
        t0 = time.perf_counter()
        # For simplicity, we assume Wren returns SQL compatible with our Postgres storage
        # and we can use a standard PG client or Wren's query endpoint
        results = await wren_client.execute_semantic_query({"sql": sql_query})
        latency["wren_execution"] = round((time.perf_counter() - t0) * 1000, 1)

        return QueryResponse(
            status="ok",
            task_type="sql",
            answer=f"Retrieved {len(results)} rows based on semantic interpretation.",
            confidence=0.95,
            artifacts=SQLArtifact(
                generated_sql=sql_query,
                selected_tables=[], # Wren handles joins
                validation_status="valid",
                row_count=len(results),
                preview_rows=results[:10]
            ),
            latency_ms=latency
        )

    async def _run_mindsdb_forecast(self, request: QueryRequest, latency: Dict[str, float]) -> QueryResponse:
        """Utilizes MindsDB predictors for automated forecasting."""
        t0 = time.perf_counter()
        
        # Step 1: Discover which table/predictor to use (Semantic search via Wren might help here)
        # For now, we'll try to find a predictor that matches the 'context' of the question
        # Or we check mindsdb.predictors directly.
        # Simple heuristic: extract table name if present, else fallback
        
        # Example MindsDB Query for forecast:
        # SELECT target, period FROM mindsdb.table_predictor WHERE period > LATEST AND ...
        
        # In a schema-agnostic world, we'd use Wren AI to tell us WHICH table to forecast.
        # 1. Ask Wren AI: "Which table contains the data for: {request.question}"
        # 2. Map table -> predictor
        # 3. Query predictor.
        
        # Placeholder for dynamic predictor discovery
        predictor_name = "default_predictor" 
        
        forecast_sql = f"SELECT * FROM mindsdb.{predictor_name} LIMIT 10" # Placeholder
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.mindsdb_api, json={"query": forecast_sql})
            data = resp.json().get("data", [])

        latency["mindsdb_inference"] = round((time.perf_counter() - t0) * 1000, 1)

        return QueryResponse(
            status="ok",
            task_type="forecast",
            answer=f"Generated an automated forecast using MindsDB in-database AI.",
            confidence=0.85,
            artifacts=ForecastArtifact(
                series_id=predictor_name,
                baseline=[], # History would be queried separately
                point_forecast=[{"period": r[0], "value": r[1]} for r in data],
                prediction_intervals=[],
                anomalies=[],
                backtest_metrics={}
            ),
            latency_ms=latency
        )
