# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env (repo root primeiro, depois backend/.env)
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    load_dotenv(os.path.join(repo_root, '.env'))
except Exception:
    pass
try:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    load_dotenv(os.path.join(backend_root, '.env'))
except Exception:
    pass

class Settings(BaseSettings):
    """
    Configurações da aplicação carregadas do ambiente.
    """
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")

    class Config:
        case_sensitive = True

settings = Settings()

try:
    if settings.DATABASE_URL:
        print(f"INFO: DATABASE_URL carregada (termina com '...{settings.DATABASE_URL[-10:]}').")
    else:
        print("AVISO: DATABASE_URL não definida.")
except Exception:
    pass