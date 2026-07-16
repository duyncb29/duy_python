from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"

    # LLM Settings
    llm_api_key: SecretStr
    llm_model: str = "gpt-4o"
    timeout_seconds: int = 30

    # API Collector Settings
    collector_base_url: str = "https://jsonplaceholder.typicode.com"
    collector_semaphore_limit: int = 10
    collector_timeout_seconds: int = 10

    # Topic 4 - Model Comparison Keys
    openrouter_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
