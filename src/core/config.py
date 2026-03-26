from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://agent:agent@db:5432/agentmcp"
    openai_api_key: str = ""
    openai_api_base: str = ""
    tavily_api_key: str = ""
    model_name: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    log_level: str = "INFO"
    log_json: bool = True
    mcp_file_reader_path: str = "src/mcp_servers/file_reader.py"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
