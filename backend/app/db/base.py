# backend/app/db/base.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings # Importa nossa config

# Cria a "Engine" - o ponto de conexão principal
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True # Verifica a conexão antes de usar
)

# Cria uma fábrica de "Sessões" (conexões individuais)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Cria uma classe Base da qual nossos modelos de tabela herdarão
Base = declarative_base()

def get_db():
    """
    Função de Dependência do FastAPI:
    Gera uma sessão com o banco, entrega para o endpoint
    e garante que ela será fechada no final.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

print("INFO: SQLAlchemy Engine e SessionLocal configurados.")