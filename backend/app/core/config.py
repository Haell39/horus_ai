# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

class Settings(BaseSettings):
    """
    Configurações da aplicação carregadas do ambiente.
    """
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    class Config:
        case_sensitive = True

settings = Settings()

print(f"INFO: DATABASE_URL carregada (termina com '...{settings.DATABASE_URL[-10:]}').")