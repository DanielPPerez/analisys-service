# src/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    trace_service_base_url: str

    class Config:
        env_file = ".env"

settings = Settings()