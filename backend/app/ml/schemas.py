# backend/app/ml/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisInput(BaseModel):
    """ Descreve a entrada esperada pelo endpoint /analyze """
    media_type: str = Field(..., pattern="^(audio|video)$", description="Tipo de mídia: 'audio' ou 'video'")
    filename: Optional[str] = None # Nome original do arquivo (informativo)

class PredictionResult(BaseModel):
    """ Descreve o resultado da previsão de um modelo """
    predicted_class: str
    confidence: float
    model_name: str

class AnalysisOutput(BaseModel):
    """ Descreve a resposta do endpoint /analyze """
    filename: Optional[str] = None
    media_type: str
    predictions: List[PredictionResult]
    ocorrencia_salva: bool = False # Indica se uma ocorrência foi gerada
    message: Optional[str] = None # Mensagem de status ou erro

print("INFO: Schemas de ML (AnalysisInput, PredictionResult, AnalysisOutput) definidos.")