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
    VIDEO_VOTE_K: int = 2  # K frames consecutivos para voto temporal (reduzido para maior sensibilidade)
    VIDEO_MOVING_AVG_M: int = 5  # janela M para média móvel de confiança
    VIDEO_THRESH_DEFAULT: float = 0.65  # default para thresholds por classe (ajustado)
    # Audio thresholds (separados) - permitem escolher audio vs video com regras distintas
    AUDIO_THRESH_DEFAULT: float = 0.60
    # Heurísticas suplementares para visão (blur/freeze/fade)
    # NOTE: many test clips show variance-of-Laplacian values in the 150-300
    # range for out-of-focus frames. The original value (80) was too low and
    # missed those cases. Increase to 300 to be more permissive in detecting
    # blur/fora_foco. You can override per-deployment via env AUDIO/VIDEO vars.
    VIDEO_BLUR_VAR_THRESHOLD: float = 518.0  # variance of Laplacian below this indicates blur (calibrated from diagnostics)
    VIDEO_MOTION_THRESHOLD: float = 2.0  # mean absolute pixel diff below this indicates freeze
    VIDEO_BRIGHTNESS_LOW: float = 50.0  # mean brightness below this indicates dark/fade
    VIDEO_BRIGHTNESS_DROP_RATIO: float = 0.5  # drop ratio between windows that indicates fade
    VIDEO_SAMPLE_RATE_HZ: float = 2.0  # default sampling rate in Hz for frame analysis
    # Edge density threshold: proportion of edge pixels (Canny) below which image
    # is considered low-detail / likely out-of-focus. Tweakable via env.
    VIDEO_EDGE_DENSITY_THRESHOLD: float = 0.015
    # Controls whether audio analysis can override the final video-based decision
    # when processing uploaded video files. Set to False to force video-only
    # decisions for visual errors (freeze/fade/fora_foco). Can be overridden
    # via environment variable VIDEO_ALLOW_AUDIO_OVERRIDE=true/false
    VIDEO_ALLOW_AUDIO_OVERRIDE: bool = False

    # When True, the backend will skip audio model loading and audio
    # analysis entirely. This ensures the system focuses only on visual
    # detections (freeze/fade/fora_foco). Default is True to make visual-only
    # mode the project default; set to False to enable audio processing.
    VIDEO_DISABLE_AUDIO_PROCESSING: bool = True

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

    def audio_thresholds(self) -> Dict[str, float]:
        """Similar to video_thresholds but reads AUDIO_THRESH_<CLASS> from env."""
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

try:
    if settings.DATABASE_URL:
        print(f"INFO: DATABASE_URL carregada (termina com '...{settings.DATABASE_URL[-10:]}').")
    else:
        print("AVISO: DATABASE_URL não definida.")
except Exception:
    pass