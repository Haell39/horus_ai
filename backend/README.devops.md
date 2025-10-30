# Horus AI — DevOps Quick Start

Este arquivo é um resumo prático para rodar e operar o backend localmente (destinado a devs/ops iniciantes).

1. Objetivo

- Iniciar/parar o backend, configurar variáveis (.env), controlar ingest SRT → HLS e verificar logs.

2. Arquivos importantes

- `backend/.env` — variáveis de ambiente (NÃO comitar). Use `backend/.env.example` como modelo.
- `backend/.env.example` — lista segura das chaves: DATABASE*URL, SRT_STREAM_URL_GLOBO, VIDEO_VOTE_K, VIDEO_MOVING_AVG_M, VIDEO_THRESH*<CLASSE>, etc.
- `backend/app/streams/srt_reader.py` — gerencia processos ffmpeg (HLS + extractor), criação de HLS e extração de frames.
- `backend/app/ml/inference.py` — carregamento dos modelos TFLite e wrappers de inferência.
- `backend/app/api/endpoints/streams.py` — endpoints: POST /api/v1/streams/start, POST /api/v1/streams/stop, GET /api/v1/streams/status.
- `backend/app/api/endpoints/ocorrencias.py` — CRUD de ocorrências (GET /api/v1/ocorrencias, POST /api/v1/ocorrencias).

3. Rodando localmente (PowerShell)

```powershell
cd backend
# ativar venv (se tiver)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# iniciar uvicorn (sem --reload para testes de streaming)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Iniciar e parar ingest (exemplo PowerShell)

```powershell
# Start
$body = @{ url = 'srt://MEU.SRT.ENDPOINT:PORT?mode=caller&passphrase=...' ; fps = 1.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body $body -ContentType 'application/json'
# Stop
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/stop
# Status
Invoke-RestMethod -Method Get -Uri http://localhost:8000/api/v1/streams/status
```

5. Logs e troubleshooting rápido

- O HLS playlist fica em `backend/static/hls/stream.m3u8` e os segmentos em `backend/static/hls/`.
- ffmpeg logs: `backend/static/hls/hls_ffmpeg.log`.
- Se o player não reproduz: verifique `stream.m3u8` e o log do ffmpeg.
- Para matar ffmpeg "presas":

```powershell
Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

6. Ajustes comuns

- Aumentar `VIDEO_VOTE_K` ou `VIDEO_MOVING_AVG_M` aumenta estabilidade das detecções (mais latência).
- Ajustar `VIDEO_THRESH_<CLASSE>` por classe para reduzir falsos positivos.

7. Segurança

- Não comite `backend/.env` com segredos. Use `backend/.env.example` para documentar as variáveis.

8. Contatos

- Para mudanças operacionais maiores (ex.: adicionar rotas de cleanup ou alterar parâmetros ffmpeg), prefira abrir uma issue/PR no repositório.
