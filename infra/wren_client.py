"""
Wren AI Enterprise Client
Integrates the backend with the Wren AI Semantic Engine and Cube.js.
Supports schema-agnostic Text-to-SQL and Semantic discovery.
"""
import httpx
import logging
import time
from typing import Any, Dict, List, Optional
from infra.settings import settings

logger = logging.getLogger(__name__)

class WrenClient:
    def __init__(self):
        # engine is on 8081 (mapped from 8080 internal)
        self.engine_url = "http://localhost:8081"
        self.ai_service_url = "http://localhost:5555"
        self.timeout = 180.0

    async def generate_sql(self, question: str, context: Optional[str] = None) -> str:
        """
        Submits a natural language question to Wren AI and returns the generated SQL.
        Wren AI is connected to Cube.js, ensuring it respects the semantic layer.
        """
        # Note: In a production setup, we first call the AI service to get the 'Intent'
        # and then the Engine to get the 'SQL'.
        
        # 1. Ask Wren AI Service for the SQL based on the semantic models (Cubes)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.ai_service_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": question}],
                        "model": "wren-ai",
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Wren usually returns the SQL or a recommendation of Cube measures/dimensions
                return data.get("sql", "")
        except Exception as e:
            logger.error(f"Wren AI SQL Generation failed: {e}")
            raise

    async def execute_semantic_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes a query directly against the Cube.js / Wren Engine.
        This provides cached and aggregated results.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.engine_url}/v1/query",
                    json=query
                )
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception as e:
            logger.error(f"Semantic query execution failed: {e}")
            return []

    async def register_datasource(self):
        """
        Registers the Postgres datasource in Wren AI. 
        Wren AI will then scan the schema and prepare for Text-to-SQL.
        """
        pass # To be implemented once Engine is healthy

wren_client = WrenClient()
