from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    gemma_base_url: str = os.getenv("GEMMA_BASE_URL", "")
    gemma_model: str = os.getenv("GEMMA_MODEL", "google/gemma-4-e2b-it")
    gemma_api_key: str = os.getenv("GEMMA_API_KEY", "")
    duckdb_path: Path = BASE_DIR / os.getenv("DUCKDB_PATH", "data/demo.duckdb")
    data_dir: Path = BASE_DIR / os.getenv("DATA_DIR", "data")
    schema_cache_path: Path = BASE_DIR / os.getenv("SCHEMA_CACHE_PATH", "data/cache/schema_cache.json")
    examples_path: Path = BASE_DIR / os.getenv("EXAMPLES_PATH", "data/examples.json")
    metric_registry_path: Path = BASE_DIR / os.getenv("METRIC_REGISTRY_PATH", "config/metric_registry.yaml")
    chroma_persist_path: Path = BASE_DIR / os.getenv("CHROMA_PERSIST_PATH", "data/cache/chroma")
    connection_profiles_path: Path = BASE_DIR / os.getenv(
        "CONNECTION_PROFILES_PATH", "data/cache/connection_profiles.json"
    )
    nim_base_url: str = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nim_model: str = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    mindsdb_api_url: str = os.getenv("MINDSDB_API_URL", "http://mindsdb:47334/api/sql/query")
    airflow_base_url: str = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
    airflow_username: str = os.getenv("AIRFLOW_USERNAME", "airflow")
    airflow_password: str = os.getenv("AIRFLOW_PASSWORD", "airflow")
    cube_base_url: str = os.getenv("CUBE_BASE_URL", "http://cubejs:4000")
    default_row_cap: int = int(os.getenv("DEFAULT_ROW_CAP", "500"))
    query_timeout_s: int = int(os.getenv("QUERY_TIMEOUT_S", "120"))
    enable_startup_refresh: bool = _env_bool("ENABLE_STARTUP_REFRESH", False)
    postgres_dsn: str = os.getenv("POSTGRES_DSN", "")
    default_connection_type: str = os.getenv("DEFAULT_CONNECTION_TYPE", "duckdb")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.schema_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_path.mkdir(parents=True, exist_ok=True)
        self.connection_profiles_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
