## Quick orientation for AI assistants and contributors

This repository is a two-part application used for ingesting live SRT streams, running ML inference (Keras/TFLite) on video and audio, persisting detections, and exposing a web UI for realtime monitoring. It contains:

- Frontend: Angular app at `frontend/` (UI & player).
- Backend: FastAPI app at `backend/app/` (API, streaming glue, ML inference, DB access).
- Static output: `backend/static/` (HLS playlist + segments in `hls/`, generated clips in `clips/`).

This document summarizes the architecture, developer workflows, important files, streaming flow (SRT → HLS → player), troubleshooting tips and recommended next steps. Keep it up to date as you change behavior in the project.

---

## Orientação rápida (copilot / contribuidores)

Este repositório é uma aplicação composta por duas partes: ingestão de streams SRT, inferência com Keras/TFLite (suporte a sequências temporais), persistência de ocorrências e uma UI Angular para monitoramento em tempo real.

- Frontend: `frontend/` — aplicação Angular 19 (player, dashboard de monitoramento, dados históricos, configurações).
- Backend: `backend/app/` — FastAPI (API REST, controle de ffmpeg, inferência ML, persistência DB, WebSocket para ocorrências).
- Saída estática: `backend/static/` — HLS (`hls/`) e clips gerados (`clips/`).

Mantenha este arquivo atualizado quando alterar rotas, flags do ffmpeg, formatos de evidência ou nomes de classes do modelo.

---

1. Visão geral rápida

- O frontend conecta ao HLS servido pelo backend em `/hls/stream.m3u8` e ao WebSocket `/ws/ocorrencias` para receber novas ocorrências em tempo real.
- O backend expõe endpoints para controlar a ingestão SRT → HLS e para CRUD de ocorrências.
- A inferência é realizada em background:
  - **Vídeo**: Buffer deslizante de N frames. Suporta modelos 4D (frame único) e 5D (sequência temporal). Usa padding (repetição de frames) para baixa latência.
  - **Áudio**: Extração via librosa com fallback para ffmpeg. Trata ausência de áudio como buffer vazio (sem crash).

2. Arquivos-chave (mapa curto)

- `backend/app/main.py` — cria a app, configura CORS, monta `StaticFiles` para `/hls` e `/clips`, carrega modelos no startup.
- `backend/app/streams/srt_reader.py` — `SRTIngestor`:
  - Gerencia processo ffmpeg (SRT → HLS + frames).
  - Usa flags tolerantes (`-err_detect ignore_err`, `+discardcorrupt`) para lidar com streams instáveis.
  - Implementa snapshotting de frames antes da geração de clips para evitar erros de I/O.
- `backend/app/api/endpoints/streams.py` — endpoints `start/stop/status` para o ingest.
- `backend/app/api/endpoints/ocorrencias.py` — listagem e criação de ocorrências (JSON).
- `backend/app/api/endpoints/ws.py` — manager de WebSocket para broadcast de novas ocorrências.
- `backend/app/ml/inference.py` — Core de ML:
  - Carrega modelos Keras/TFLite.
  - Detecta input shapes (4D vs 5D) e gerencia filas de sequência (`deque`).
  - Implementa fallback de áudio via ffmpeg subprocess.
  - Calcula heurísticas (blur, brilho, movimento).
- `backend/app/core/config.py` — carrega variáveis de ambiente.
- `frontend/src/app/pages/` — Componentes principais: `Monitoramento` (Live), `Dados` (Dashboards), `Configuracoes`, `Cortes`.

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

- `backend/.env` deve conter:
  - `VIDEO_VOTE_K` (K frames consecutivos para voto)
  - `VIDEO_MOVING_AVG_M` (janela M para média móvel)
  - `VIDEO_THRESH_<CLASSE>` por classe (ex.: `VIDEO_THRESH_BORRADO=0.7`)
  - `VIDEO_THRESH_DEFAULT` fallback

6. Troubleshooting rápido

- Logs do ffmpeg: `backend/static/hls/hls_ffmpeg.log`.
- Playlist HLS: `backend/static/hls/stream.m3u8`.
- **Erros H.264**: Mensagens como "Missing reference picture" ou "non-existing PPS" são comuns em streams SRT instáveis. O backend usa flags para ignorar erros não fatais e continuar o processamento.
- **Áudio**: Se o log mostrar "AVISO: Áudio completo vazio", significa que o ffmpeg não encontrou stream de áudio ou falhou na extração. O sistema segue operando sem análise de áudio.
- Matando ffmpeg pendente (PowerShell): `Get-Process -Name ffmpeg | Stop-Process -Force`.

7. Próximos passos sugeridos

- Adicionar retry/backoff em `SRTIngestor.start()` (meio-ambiente instável SRT).
- Expor configuração de política de padding (repetir vs esperar) via env var.
- Expor configuração de comportamento "sem áudio" (ignorar vs gerar alerta) via env var.
- Endpoint administrativo `/api/v1/streams/cleanup` para kill+cleanup seguro.

---

Se fizerem atualizações no frontend ou mudarem os nomes das classes no modelo, atualize também as variáveis `VIDEO_THRESH_<CLASSE>` e este documento.
