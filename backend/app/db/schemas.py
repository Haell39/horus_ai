from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

class OcorrenciaBase(BaseModel):
    """Campos comuns para Create e Read."""
    start_ts: datetime
    end_ts: datetime
    duration_s: Optional[float] = None
    category: Optional[str] = None
    type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None

class OcorrenciaCreate(OcorrenciaBase):
    """Schema para criar ocorrência."""
    pass

class OcorrenciaRead(OcorrenciaBase):
    """Schema para leitura com campos gerados (id, created_at)."""
    id: int
    created_at: datetime

    class Config:
        from_attributes = True