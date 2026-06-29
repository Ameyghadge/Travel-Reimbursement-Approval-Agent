"""Centralized configuration for the Travel Reimbursement Agent."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env with sensible defaults."""

    # LLM
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Inference
    max_new_tokens: int = 300
    temperature: float = 0.0
    max_retries: int = 3

    # RAG
    top_k_chunks: int = 2
    chunk_size: int = 512
    chunk_overlap: int = 50
    policy_path: str = "data/travel_policy.md"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
