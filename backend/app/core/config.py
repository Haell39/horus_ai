import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Dict

# Carrega .env
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    load_dotenv(os.path.join(repo_root, '.env'))
except Exception:
    pass

class Settings(BaseSettings):
    """Configurações carregadas do ambiente."""
    DATABASE_URL: str | None = os.getenv("DATABASE_URL")
    
    # Tuning de vídeo
    VIDEO_VOTE_K: int = 2
    VIDEO_MOVING_AVG_M: int = 5
    VIDEO_THRESH_DEFAULT: float = 0.65
    AUDIO_THRESH_DEFAULT: float = 0.60
    
    # Heurísticas de visão
    VIDEO_BLUR_VAR_THRESHOLD: float = 518.0
    VIDEO_MOTION_THRESHOLD: float = 2.0
    VIDEO_BRIGHTNESS_LOW: float = 50.0
    VIDEO_BRIGHTNESS_DROP_RATIO: float = 0.5
    VIDEO_SAMPLE_RATE_HZ: float = 2.0
    VIDEO_EDGE_DENSITY_THRESHOLD: float = 0.015
    VIDEO_ALLOW_AUDIO_OVERRIDE: bool = False
    VIDEO_DISABLE_AUDIO_PROCESSING: bool = True

    def video_thresholds(self) -> Dict[str, float]:
        """Retorna thresholds por classe de VIDEO_THRESH_<CLASSE> do ambiente."""
        thresh: Dict[str, float] = {}
        for k, v in os.environ.items():
            if not k.startswith("VIDEO_THRESH_"):
                continue
            name = k[len("VIDEO_THRESH_"):]
            try:
                thresh[name] = float(v)
            except Exception:
                continue
        if not thresh:
            thresh["DEFAULT"] = float(self.VIDEO_THRESH_DEFAULT)
        return thresh

    def audio_thresholds(self) -> Dict[str, float]:
        """Retorna thresholds por classe de AUDIO_THRESH_<CLASSE> do ambiente."""
        thresh: Dict[str, float] = {}
        for k, v in os.environ.items():
            if not k.startswith("AUDIO_THRESH_"):
                continue
            name = k[len("AUDIO_THRESH_"):]
            try:
                thresh[name] = float(v)
            except Exception:
                continue
        if not thresh:
            thresh["DEFAULT"] = float(self.AUDIO_THRESH_DEFAULT)
        return thresh

    class Config:
        case_sensitive = True

settings = Settings()