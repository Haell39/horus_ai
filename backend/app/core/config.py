# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Dict

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
    # Video detection tuning
    VIDEO_VOTE_K: int = 3  # K frames consecutivos para voto temporal
    VIDEO_MOVING_AVG_M: int = 5  # janela M para média móvel de confiança
    VIDEO_THRESH_DEFAULT: float = 0.7  # default para thresholds por classe

    def video_thresholds(self) -> Dict[str, float]:
        """Retorna um dicionário com thresholds por classe extraídos das
        variáveis de ambiente que começam com `VIDEO_THRESH_`.

        Ex.: env `VIDEO_THRESH_BORRADO=0.7` -> {'BORRADO': 0.7}
        Se nenhum threshold por classe for encontrado, retorna um mapping
        com a chave 'DEFAULT' apontando para `VIDEO_THRESH_DEFAULT`.
        """
        thresh: Dict[str, float] = {}
        for k, v in os.environ.items():
            if not k.startswith("VIDEO_THRESH_"):
                continue
            name = k[len("VIDEO_THRESH_"):]
            try:
                thresh[name] = float(v)
            except Exception:
                # ignore malformed values
                continue
        if not thresh:
            thresh["DEFAULT"] = float(self.VIDEO_THRESH_DEFAULT)
        return thresh

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