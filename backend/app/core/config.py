"""Loads environment variables and application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Resolve the project root so .env can be loaded
# regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Central application configuration, populated from environment/.env."""

    model_config = SettingsConfigDict(
    env_file=ENV_FILE,
    extra="ignore",
    )

    # PostgreSQL connection
    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/mindcompass"

    # OpenAI credentials
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o"

    # Authentication settings
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # ChromaDB settings
    chroma_persist_directory: str = "./chroma_data"
    chroma_collection_name: str = "agentic_ai_materials"

    # Backend URL used by Streamlit
    fastapi_base_url: str = "http://localhost:8000"

    # RAG ingestion / chunking (new, additive)
    knowledge_base_source_dir: str = ""
    rag_chunk_size: int = 500
    rag_chunk_overlap_ratio: float = 0.15  # 15% overlap, within the 10-20% target


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""

    return Settings()
