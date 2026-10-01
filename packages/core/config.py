"""Central configuration.

Every infrastructure dependency has a production integration *and* an embedded fallback so the
platform can be demonstrated on a laptop without Docker:

| Concern        | Production (Docker / on-prem)        | Embedded fallback (empty URL)          |
|----------------|--------------------------------------|----------------------------------------|
| SQL            | PostgreSQL / Microsoft SQL Server    | SQLite file                            |
| Vector search  | Qdrant server                        | qdrant-client local mode (same API)    |
| Lexical search | Elasticsearch 8 (BM25)               | In-process BM25 index                  |
| Cache / queue  | Redis                                | In-process dict / queue                |
| LLM            | vLLM (OpenAI-compatible) / OpenAI    | Deterministic mock provider            |
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "docker", "test", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Data stores -------------------------------------------------------------------------
    database_url: str = f"sqlite:///{(ROOT_DIR / '.data' / 'novapromo.db').as_posix()}"
    # Separate connection for the controlled analytics-SQL mode. In Docker this is a
    # PostgreSQL role with SELECT-only grants on an allowlisted set of tables.
    analytics_database_url: str = ""
    qdrant_url: str = ""  # empty => embedded local mode
    qdrant_local_path: str = str(ROOT_DIR / ".data" / "qdrant")
    elasticsearch_url: str = ""  # empty => in-process BM25
    redis_url: str = ""  # empty => in-process cache/queue
    data_dir: str = str(ROOT_DIR / "data")

    # --- LLM ---------------------------------------------------------------------------------
    llm_provider: Literal["mock", "openai", "vllm"] = "mock"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700
    llm_timeout_s: float = 60.0
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    vllm_api_key: str = "EMPTY"
    # Use the LLM (tool calling) for routing when a real provider is configured.
    llm_routing: bool = False

    # --- Embeddings / reranking --------------------------------------------------------------
    embedding_provider: Literal["hash", "openai_compatible", "fastembed"] = "hash"
    embedding_base_url: str = "http://localhost:8002/v1"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_api_key: str = "EMPTY"
    embedding_dim: int = 512
    reranker_provider: Literal["feature", "cross_encoder", "http"] = "feature"
    reranker_url: str = "http://localhost:8003/rerank"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Search ------------------------------------------------------------------------------
    rrf_k: int = 60
    search_candidates: int = 50
    products_index: str = "novapromo-products"
    products_collection: str = "products"
    documents_collection: str = "documents"

    # --- Security ----------------------------------------------------------------------------
    jwt_secret: str = Field(default="change-me-in-production-please-32b", min_length=16)
    jwt_ttl_minutes: int = 480
    demo_password: str = "demo"

    # --- Analytics SQL -----------------------------------------------------------------------
    analytics_sql_enabled: bool = True
    analytics_row_limit: int = 200
    analytics_timeout_ms: int = 3000

    # --- Batch -------------------------------------------------------------------------------
    batch_mode: Literal["inline", "worker"] = "inline"
    batch_max_attempts: int = 3
    batch_concurrency: int = 4
    batch_checkpoint_every: int = 25
    batch_schedule_hour: int = 2  # nightly run at 02:00 local time (worker mode)
    batch_auto_fill_missing: bool = True  # AI may fill EMPTY fields; overwrites need approval

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def effective_analytics_url(self) -> str:
        return self.analytics_database_url or self.database_url

    @property
    def active_model(self) -> str:
        return {"mock": "mock-deterministic-v1", "openai": self.openai_model, "vllm": self.vllm_model}[
            self.llm_provider
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
