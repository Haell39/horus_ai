## Quick orientation for AI assistants and contributors

This repository is a two-part application used for ingesting live SRT streams, running lightweight ML inference (TFLite), persisting detections, and exposing a web UI for realtime monitoring. It contains:

- Frontend: Angular app at `frontend/` (UI & player).
- Backend: FastAPI app at `backend/app/` (API, streaming glue, ML inference, DB access).
- Static output: `backend/static/` (HLS playlist + segments in `hls/`, generated clips in `clips/`).

This document summarizes the architecture, developer workflows, important files, streaming flow (SRT → HLS → player), troubleshooting tips and recommended next steps. Keep it up to date as you change behavior in the project.

---

## Orientação rápida (copilot / contribuidores)

Este repositório é uma aplicação composta por duas partes: ingestão de streams SRT, inferência leve com TFLite, persistência de ocorrências e uma UI Angular para monitoramento em tempo real.

- Frontend: `frontend/` — aplicação Angular (player, início/parada de ingest, exibe ocorrências e clips).
- Backend: `backend/app/` — FastAPI (API REST, controle de ffmpeg, inferência TFLite, persistência DB, WebSocket para ocorrências).
- Saída estática: `backend/static/` — HLS (`hls/`) e clips gerados (`clips/`).

Mantenha este arquivo atualizado quando alterar rotas, flags do ffmpeg, formatos de evidência ou nomes de classes do modelo.

---

1. Visão geral rápida

- O frontend conecta ao HLS servido pelo backend em `/hls/stream.m3u8` e ao WebSocket `/ws/ocorrencias` para receber novas ocorrências em tempo real.
- O backend expõe endpoints para controlar a ingestão SRT → HLS e para CRUD de ocorrências. A inferência por frame é feita por um watcher que extrai frames com ffmpeg e usa TFLite para decisão por frame.

2. Arquivos-chave (mapa curto)

- `backend/app/main.py` — cria a app, configura CORS, monta `StaticFiles` para `/hls` e `/clips`, carrega modelos no startup e loga configs relevantes.
- `backend/app/streams/srt_reader.py` — `SRTIngestor`: gerencia ffmpeg (HLS + extractor), roda loop de inferência por frame, cria clips e persiste ocorrências.
- `backend/app/api/endpoints/streams.py` — endpoints `start/stop/status` para o ingest.
- `backend/app/api/endpoints/ocorrencias.py` — listagem e criação de ocorrências (JSON).
- `backend/app/api/endpoints/ws.py` — manager de WebSocket para broadcast de novas ocorrências.
- `backend/app/ml/inference.py` — wrappers TFLite para áudio e vídeo (`run_video_topk`, `analyze_video_frames`, etc.).
- `backend/app/core/config.py` — carrega variáveis de ambiente, agora com defaults tipados e helper para `VIDEO_THRESH_*`.
- `backend/.env.example` — arquivo exemplo (safe) com as variáveis esperadas.

3. Como rodar (dev local, PowerShell)

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
cd ../frontend
npm install
npm start
```

4. Endpoints importantes

- POST `/api/v1/streams/start` { url, fps? }
- POST `/api/v1/streams/stop`
- GET `/api/v1/streams/status`
- GET `/api/v1/ocorrencias` (JSON list)
- WS `/ws/ocorrencias` (broadcast em tempo real)

5. Configuração de tuning (env)

- `backend/.env` deve conter (exemplos):
  - `VIDEO_VOTE_K` (K frames consecutivos para voto)
  - `VIDEO_MOVING_AVG_M` (janela M para média móvel)
  - `VIDEO_THRESH_<CLASSE>` por classe (ex.: `VIDEO_THRESH_BORRADO=0.7`)
  - `VIDEO_THRESH_DEFAULT` fallback

6. Troubleshooting rápido

- Logs do ffmpeg: `backend/static/hls/hls_ffmpeg.log`.
- Playlist HLS: `backend/static/hls/stream.m3u8`.
- Matando ffmpeg pendente (PowerShell): `Get-Process -Name ffmpeg | Stop-Process -Force`.

7. Próximos passos sugeridos

- Adicionar retry/backoff em `SRTIngestor.start()` (meio-ambiente instável SRT).
- Endpoint administrativo `/api/v1/streams/cleanup` para kill+cleanup seguro.
- Endpoint read-only `/api/v1/config` (apenas non-sensitive) para ops verificarem os valores ativos.

---

Se quiser, eu adapto este documento para um README conciso para DevOps ou para Product (resumido). Se fizerem atualizações no frontend ou mudarem os nomes das classes no modelo, atualize também as variáveis `VIDEO_THRESH_<CLASSE>` e este documento.

- POST /api/v1/streams/start { url, fps? }
