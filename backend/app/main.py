# backend/app/main.py
# (Versão Completa com Rota de Análise Incluída)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importa NOSSOS routers
from app.api.endpoints import ocorrencias
from app.api.endpoints import analysis # <<< ESSA LINHA É CRUCIAL
from app.api.endpoints import ws
from app.api.endpoints import streams
from fastapi.staticfiles import StaticFiles
import os

# Importa a base do banco para criação de tabelas
from app.db.base import Base, engine
# Garante carregamento do .env e configurações logo no startup
from app.core.config import settings  # noqa: F401

# --- Criação das Tabelas ---
print("INFO: Verificando e criando tabelas no banco de dados (se necessário)...")
try:
    Base.metadata.create_all(bind=engine)
    print("INFO: Tabelas OK.")
except Exception as e:
    print(f"ERRO: Não foi possível conectar ou criar tabelas: {e}")
    # Considerar parar a aplicação aqui em caso de erro crítico de DB

# --- Instância Principal do FastAPI ---
app = FastAPI(
    title="Horus AI - Backend",
    description="API para monitoramento, análise e registro de ocorrências da Globo.",
    version="0.1.0"
)

# --- Configuração do CORS ---
# Permite que seu frontend (ex: localhost:4200) acesse esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em prod, restrinja para o domínio do seu frontend
    allow_credentials=True,
    allow_methods=["*"], # Permite GET, POST, PUT, DELETE, etc.
    allow_headers=["*"], # Permite todos os cabeçalhos comuns
)


# Some browsers (and certain fetch usages) may still hit StaticFiles without
# receiving the CORS headers in some environments; add a small middleware that
# ensures static HLS/CLIPS responses contain an Access-Control-Allow-Origin
# header so the frontend can load .m3u8/.ts segments from another origin.
@app.middleware("http")
async def ensure_static_cors(request, call_next):
    response = await call_next(request)
    try:
        path = request.url.path or ''
        if path.startswith('/hls') or path.startswith('/clips'):
            # be permissive for local dev; in prod scope this to your frontend origin
            response.headers['Access-Control-Allow-Origin'] = '*' 
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
            # Ensure HLS playlists/segments are not cached by the browser so the
            # player always requests the latest playlist/segments.
            if path.startswith('/hls'):
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    except Exception:
        pass
    return response

# --- Inclusão dos Routers ---

# Router para Ocorrências (CRUD)
app.include_router(
    ocorrencias.router,
    prefix="/api/v1",       # Prefixo comum para a API
    tags=["Ocorrências"]    # Agrupa na documentação /docs
)

# <<< BLOCO ADICIONADO PARA O ROUTER DE ANÁLISE ML >>>
app.include_router(
    analysis.router,
    prefix="/api/v1",       # Mesmo prefixo da API
    tags=["Análise ML"]     # Tag separada na documentação
)
# <<< FIM DO BLOCO ADICIONADO >>>

app.include_router(
    ws.router,
    # Nota: WebSockets não costumam ter prefixo /api/v1
    # O endpoint será /ws/ocorrencias
    tags=["WebSockets"]
)
# Streams control endpoints (start/stop)
app.include_router(
    streams.router,
    prefix="/api/v1",
    tags=["Streams"]
)
# <<< FIM DO BLOCO ADICIONADO >>>

# --- Endpoint Raiz (Saúde) ---
@app.get("/", tags=["Root"])
def read_root():
    """Verifica se a API está online."""
    return {"message": "Horus AI API - Online"}

print("INFO: Aplicação FastAPI iniciada e rotas configuradas.")

# --- Monta rota para servir clipes estáticos (ex: /clips/clip_123.mp4) ---
clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'clips'))
os.makedirs(clips_dir, exist_ok=True)
app.mount("/clips", StaticFiles(directory=clips_dir), name="clips")
print(f"INFO: Static clips mount configured at /clips -> {clips_dir}")

# HLS mount (stream output)
hls_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'hls'))
os.makedirs(hls_dir, exist_ok=True)
app.mount("/hls", StaticFiles(directory=hls_dir), name="hls")
print(f"INFO: HLS mount configured at /hls -> {hls_dir}")

# === Opcional: Evento Startup para Carregar Modelos ===
# (Carrega modelos na inicialização do servidor)
@app.on_event("startup")
async def startup_event():
    print("INFO: Evento startup - Carregando modelos de ML...")
    # Importa DENTRO da função para garantir que tudo esteja pronto
    from app.ml import inference
    # Chama a função load_all_models apenas se não foram carregados ainda
    if not inference.models_loaded:
         inference.load_all_models() # Tenta carregar

    if not inference.models_loaded:
        print("ALERTA CRÍTICO: Modelos de ML não puderam ser carregados no startup! Endpoint /analyze pode falhar.")
    else:
        print("INFO: Modelos de ML verificados/carregados no startup.")
# === Fim do Bloco Opcional ===