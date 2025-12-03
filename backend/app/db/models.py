from sqlalchemy import Column, Integer, DateTime, Float, Text, JSON
from sqlalchemy.sql import func
from .base import Base

class Ocorrencia(Base):
    """Modelo ORM para tabela 'globo.ocorrencias'."""
    __tablename__ = "ocorrencias"
    __table_args__ = {'schema': 'globo'}

    id = Column(Integer, primary_key=True, index=True)
    start_ts = Column(DateTime(timezone=True), nullable=False)
    end_ts = Column(DateTime(timezone=True), nullable=False)
    duration_s = Column(Float, nullable=True)
    category = Column(Text, nullable=True)
    type = Column(Text, nullable=True)
    severity = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Ocorrencia(id={self.id}, type='{self.type}')>"