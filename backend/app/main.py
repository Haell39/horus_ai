# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importa nossos routers e a base do banco
from app.api.endpoints import ocorrencias
from app.db.base import Base, engine

# --- Criação das Tabelas ---
# (Em dev, podemos criar as tabelas aqui se não existirem)
# (Em prod, usaríamos Alembic para migrações)
print("INFO: Verificando e criando tabelas no banco de dados (se necessário)...")
try:
    Base.metadata.create_all(bind=engine)
    print("INFO: Tabelas OK.")
except Exception as e:
    print(f"ERRO: Não foi possível conectar ou criar tabelas: {e}")
    # Em um app real, talvez quiséssemos parar aqui.

# --- Instância Principal do FastAPI ---
app = FastAPI(
    title="Horus AI - Backend",
    description="API para monitoramento e registro de ocorrências da Globo.",
    version="0.1.0"
)

# --- Configuração do CORS ---
# Permite que seu frontend (ex: localhost:4200) acesse esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em prod, mude para ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"], # Permite GET, POST, etc.
    allow_headers=["*"],
)

# --- Inclusão dos Routers ---
# Adiciona os endpoints de /ocorrencias com o prefixo /api/v1
app.include_router(
    ocorrencias.router,
    prefix="/api/v1",
    tags=["Ocorrências"]
)

# --- Endpoint Raiz (Saúde) ---
@app.get("/", tags=["Root"])
def read_root():
    """Verifica se a API está online."""
    return {"message": "Horus AI API - Online"}

print("INFO: Aplicação FastAPI iniciada e rotas configuradas.")