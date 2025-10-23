# backend/app/db/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class OcorrenciaBase(BaseModel):
    """
    Campos comuns, compartilhados por 'Create' e 'Read'.
    """
    start_ts: datetime
    end_ts: datetime
    duration_s: Optional[float] = None
    category: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None # Aceita um JSON


class OcorrenciaCreate(OcorrenciaBase):
    """
    Schema para CRIAR uma nova ocorrência (o que a API recebe).
    Herda todos os campos de OcorrenciaBase.
    """
    pass


class OcorrenciaRead(OcorrenciaBase):
    """
    Schema para LER uma ocorrência (o que a API retorna).
    Inclui os campos gerados pelo banco (id, created_at).
    """
    id: int
    created_at: datetime

    class Config:
        from_attributes = True # Mágica: permite o Pydantic ler dados de um objeto SQLAlchemy

print("INFO: Schemas Pydantic (OcorrenciaBase, Create, Read) definidos.")