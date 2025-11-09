# backend/app/main.py
# (Versão Completa com Rota de Análise Incluída)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Importa NOSSOS routers
from app.api.endpoints import ocorrencias
from app.api.endpoints import analysis # <<< ESSA LINHA É CRUCIAL
from app.api.endpoints import docs
from app.api.endpoints import ws
from app.api.endpoints import streams
from app.api.endpoints import admin
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

# Docs viewer endpoints (lists markdown files under repo folders)
app.include_router(
    docs.router,
    prefix="/api/v1",
    tags=["Docs"]
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
app.include_router(
    admin.router,
    prefix="/api/v1",
    tags=["Admin"]
)
# <<< FIM DO BLOCO ADICIONADO >>>

# --- Endpoint Raiz (Saúde) ---
@app.get("/", tags=["Root"])
def read_root():
    """Verifica se a API está online."""
    return {"message": "Horus AI API - Online"}

print("INFO: Aplicação FastAPI iniciada e rotas configuradas.")

# --- Monta rota para servir clipes estáticos (ex: /clips/clip_123.mp4) ---
# Monta a pasta pública padrão para clipes
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
    # --- Log não-sensível das configurações relevantes para vídeo ---
    try:
        # settings está importado no topo do arquivo
        v_vote = settings.VIDEO_VOTE_K
        v_mov = settings.VIDEO_MOVING_AVG_M
        v_thresh_default = settings.VIDEO_THRESH_DEFAULT
        v_thresh_map = settings.video_thresholds()
        print(f"INFO: VIDEO_VOTE_K={v_vote}, VIDEO_MOVING_AVG_M={v_mov}, VIDEO_THRESH_DEFAULT={v_thresh_default}")
        # list a few per-class thresholds (limit output to 20 entries)
        sample = list(v_thresh_map.items())[:20]
        if sample:
            print("INFO: VIDEO_THRESH (sample): " + ", ".join(f"{k}:{v}" for k, v in sample))
    except Exception as _:
        # do not crash startup on logging
        pass
# === Fim do Bloco Opcional ===