import os
import json
import httpx
from vanna.legacy.base.base import VannaBase
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore

from infra.settings import settings
from infra.semantic_layer import semantic_layer

class NIM_Vanna(ChromaDB_VectorStore, VannaBase):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        VannaBase.__init__(self, config=config)
        self.api_key = settings.nvidia_api_key
        self.base_url = settings.nim_base_url.rstrip("/")
        self.model = settings.nim_model

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        """Synchronously submit the prompt to NIM Gateway."""
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured")
            
        # Vanna sometimes passes a string or a list of dicts.
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", 0.0),
                    "max_tokens": kwargs.get("max_tokens", 500),
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
            
    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """Override to inject Semantic layer knowledge into the Vanna prompt."""
        # Dynamically inject the schema DDL so Vanna doesn't hallucinate columns
        from orchestrator.graph import AppServices
        services = AppServices.bootstrap()
        for table, details in services.schema_cache.get("details", {}).items():
            cols = [f"{c['name']} {c['type']}" for c in details.get("columns", []) if isinstance(c, dict)]
            if cols:
                ddl_list.append(f"CREATE TABLE {table} (\n  " + ",\n  ".join(cols) + "\n);")
                
        # Call Vanna's base implementation which finds matching DDL and Schema
        rendered_prompt = super().get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs
        )
        
        # Inject our semantic business knowledge!
        semantic_context = semantic_layer.get_context_for_sql()
        
        # We append the context to the end or beginning of the prompt.
        if isinstance(rendered_prompt, list):
            # Vanna usually returns a list of dictionaries for chat models.
            # We inject the semantic context into the system message.
            for msg in rendered_prompt:
                if msg.get("role") == "system":
                    msg["content"] += f"\n\nBUSINESS LOGIC (Priority Rules):\n{semantic_context}\n\nIMPORTANT: Treat ANY `_forecast` table as the standard table when predicting future dates.\nCRITICAL RULE: You are a SQL generator. DO NOT REFUSE or say you cannot provide exact future dates. Future dates are explicitly available in the `_forecast` tables. Output ONLY valid SQL."
                    break
        elif isinstance(rendered_prompt, str):
            rendered_prompt += f"\n\nBUSINESS LOGIC:\n{semantic_context}"
            
        return rendered_prompt


# Global Vanna Instance
def init_vanna():
    settings.chroma_persist_path.mkdir(parents=True, exist_ok=True)
    
    vn = NIM_Vanna(config={'path': str(settings.chroma_persist_path)})
    
    # We use DuckDB for running the queries
    vn.connect_to_duckdb(url=str(settings.duckdb_path))
    return vn

vn = init_vanna()
