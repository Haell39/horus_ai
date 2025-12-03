from fastapi import FastAPI
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import ocorrencias, analysis, docs, ws, streams, admin, ml_info
from fastapi.staticfiles import StaticFiles
import os
from app.core import storage as storage_core
from app.db.base import Base, engine
from app.core.config import settings

print("INFO: Criando tabelas no banco de dados...")
try:
    Base.metadata.create_all(bind=engine)
    print("INFO: Tabelas OK.")
except Exception as e:
    print(f"ERRO: Falha ao criar tabelas: {e}")

app = FastAPI(
    title="Horus AI - Backend",
    description="API para monitoramento e análise de ocorrências.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_static_cors(request, call_next):
    """Garante headers CORS para arquivos estáticos HLS/clips."""
    response = await call_next(request)
    try:
        path = request.url.path or ''
        if path.startswith('/hls') or path.startswith('/clips'):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = '*'
            if path.startswith('/hls'):
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    except Exception:
        pass
    return response

app.include_router(ocorrencias.router, prefix="/api/v1", tags=["Ocorrências"])
app.include_router(analysis.router, prefix="/api/v1", tags=["Análise ML"])
app.include_router(ml_info.router, prefix="/api/v1", tags=["ML Info"])
app.include_router(docs.router, prefix="/api/v1", tags=["Docs"])
app.include_router(ws.router, tags=["WebSockets"])
app.include_router(streams.router, prefix="/api/v1", tags=["Streams"])
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Horus AI API - Online"}

print("INFO: Rotas configuradas.")

try:
    clips_dir = storage_core.get_clips_dir()
except Exception:
    clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'clips'))
os.makedirs(clips_dir, exist_ok=True)
app.mount("/clips", StaticFiles(directory=clips_dir), name="clips")

hls_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'hls'))
os.makedirs(hls_dir, exist_ok=True)
app.mount("/hls", StaticFiles(directory=hls_dir), name="hls")

@app.on_event("startup")
async def startup_event():
    """Carrega modelos ML e configura exception handler."""
    from app.ml import inference
    if not inference.models_loaded:
        inference.load_all_models()
    
    if not inference.models_loaded:
        print("ALERTA: Modelos de ML não carregados!")
    else:
        print("INFO: Modelos ML carregados.")
    
    # Log de configurações
    try:
        print(f"INFO: VIDEO_VOTE_K={settings.VIDEO_VOTE_K}, VIDEO_MOVING_AVG_M={settings.VIDEO_MOVING_AVG_M}")
    except Exception:
        pass
    
    # Handler para ignorar ConnectionResetError (cliente desconectou)
    try:
        loop = asyncio.get_running_loop()
        def _asyncio_exception_handler(loop, context):
            ex = context.get('exception')
            if isinstance(ex, ConnectionResetError):
                return
            if isinstance(ex, OSError) and getattr(ex, 'winerror', None) == 10054:
                return
            try:
                loop.default_exception_handler(context)
            except Exception:
                print("ERROR:", context)
        loop.set_exception_handler(_asyncio_exception_handler)
    except Exception:
        pass