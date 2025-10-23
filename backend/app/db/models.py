# backend/app/db/models.py
from sqlalchemy import Column, Integer, DateTime, Float, Text, JSON
from sqlalchemy.sql import func
from .base import Base # Importa a Base que criamos

class Ocorrencia(Base):
    """
    Modelo ORM que representa a tabela 'ocorrencias' no schema 'globo'.
    """
    # Nome da tabela e do schema
    __tablename__ = "ocorrencias"
    __table_args__ = {'schema': 'globo'}

    # Mapeamento exato das colunas que você definiu
    id = Column(Integer, primary_key=True, index=True)
    start_ts = Column(DateTime(timezone=True), nullable=False)
    end_ts = Column(DateTime(timezone=True), nullable=False)
    duration_s = Column(Float, nullable=True)
    category = Column(Text, nullable=True)
    type = Column(Text, nullable=True)
    severity = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    evidence = Column(JSON, nullable=True) # jsonb do Postgres vira JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Ocorrencia(id={self.id}, type='{self.type}')>"

print(f"INFO: Modelo 'Ocorrencia' mapeado para '{Ocorrencia.__table_args__['schema']}.{Ocorrencia.__tablename__}'.")