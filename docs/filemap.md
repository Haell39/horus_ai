<!-- Arquivo gerado automaticamente: docs/filemap.md -->

# FILE MAP — Horus AI (descritivo completo)

Objetivo: mapa legível do repositório com a função de cada arquivo/pasta para facilitar onboarding e manutenção.

Formato: caminho — <tipo> — descrição curta.

---

## ROOT

- README.md — documento principal do projeto (Quick Start, visão geral).
- FILEMAP.md — mapa detalhado do repositório (versão raiz).
- backend/ — backend FastAPI + ML + streaming glue.
- frontend/ — Angular app (UI, player, páginas).

## backend/

- requirements.txt — dependências Python.
- .env.example — arquivo de exemplo com variáveis de ambiente (safe) para copiar para `backend/.env`.
- .env (local) — NÃO COMMITAR; contém DATABASE_URL, SRT URL, thresholds, etc.

## backend/app/

- **init**.py
- main.py — inicializa FastAPI, configura CORS, monta `StaticFiles` (/hls, /clips), inclui routers, carrega modelos no startup e loga valores de config.

## backend/app/core/

- **init**.py
- config.py — gerencia carregamento de variáveis de ambiente via pydantic/base settings, agora com:
  - VIDEO_VOTE_K, VIDEO_MOVING_AVG_M, VIDEO_THRESH_DEFAULT (defaults e tipos)
  - helper `video_thresholds()` que agrega `VIDEO_THRESH_*` do ambiente em um dicionário.

## backend/app/api/

- endpoints/
  - **init**.py — agrega routers exportados
  - streams.py — endpoints para controlar ingest (POST /api/v1/streams/start, POST /api/v1/streams/stop, GET /api/v1/streams/status). Chamam `SRTIngestor`.
  - ocorrencias.py — CRUD simples de ocorrências (GET /api/v1/ocorrencias, POST /api/v1/ocorrencias).
  - analysis.py — endpoint de upload/analysis (exemplo de BackgroundTasks e processamento via `inference.py`).
  - ws.py — WebSocket manager para broadcast de novas ocorrências em tempo real (/ws/ocorrencias).

## backend/app/db/

- base.py — sessão SQLAlchemy (SessionLocal), engine, helper `get_db()` usado por dependências FastAPI.
- models.py — modelos SQLAlchemy (Ocorrencia com fields: id, start_ts, end_ts, duration_s, category, type, severity, confidence, evidence, created_at).
- schemas.py — Pydantic schemas (OcorrenciaCreate, OcorrenciaRead) para request/response validation.

## backend/app/ml/

- **init**.py
- inference.py — wrapper de inferência TFLite:
  - caminhos de modelos em `models/`
  - `load_all_models()` carrega intérpretes TFLite
  - helpers: `run_tflite_inference`, `run_tflite_inference_topk`, `run_video_topk`, `run_video_inference`, `analyze_video_frames`, `analyze_audio_segments`.
  - define AUDIO_CLASSES e VIDEO_CLASSES (ex.: `VIDEO_CLASSES = ['bloco','borrado','normal']`).
- models/ — pasta com arquivos `.tflite` (e.g. `video_model_quant.tflite`, `audio_model_quant.tflite`).

## backend/app/streams/

- srt_reader.py — classe `SRTIngestor`:
  - monta comandos ffmpeg para HLS (remux) e para extrair frames para inferência
  - lógica de start(): cria tmpdir, inicia processo HLS, espera playlist, inicia extractor de frames, inicia watcher thread
  - watcher (`_watch_loop`) lê frames, chama `inference.run_video_topk`, aplica voto temporal e média móvel, gera clips (via ffmpeg frame sequence), persiste ocorrências no banco e broadcast via WS
  - stop(): termina subprocessos, fecha logs, limpa `backend/static/hls/` e escreve `#EXT-X-ENDLIST` no playlist

## backend/static/

- hls/ — saída HLS: `stream.m3u8` + segmentos `.ts` + `hls_ffmpeg.log` (transientes — ignorado pelo git)
- clips/ — clips gerados por eventos (mp4) — servidos via `/clips` static mount

## backend/README.devops.md

- instruções rápidas para ops: start/stop, .env, logs, troubleshooting (criado por mim durante esta sessão)

## frontend/

- package.json — dependências & scripts (npm start => ng serve)
- angular.json, tsconfig\*.json — build configs
- src/
  - index.html, main.ts, styles.css
  - app/
    - app.component.\* — bootstrap app
    - app.routes.ts, app.config.ts
    - pages/
      - monitoramento/
        - monitoramento.component.ts — lógica do player (hls.js attach, playlist polling), Start/Stop UI, reconexão/backoff, lista de ocorrências
        - monitoramento.component.html/.css — UI
      - dados/, cortes/, configuracoes/ — outras páginas da UI
    - services/
      - ocorrencia.service.ts — chamada a `/api/v1/ocorrencias` e streams endpoints
      - websocket.service.ts — conexão WS para `/ws/ocorrencias`
    - models/
      - ocorrencia.ts — interface TypeScript para ocorrência

## .gitignore

- deve incluir `backend/static/hls/`, `backend/.env`, `node_modules/`, `.venv/` etc. (verifique se `.env` e hls estão ignorados)

## .scripts / misc

- não há scripts automatizados adicionais atualmente; podemos adicionar scripts PowerShell para criar `backend/.env` a partir do exemplo.

---

## Notas operacionais / dicas rápidas

- Para reduzir logspam no console, inicie uvicorn com `--log-level warning` ou ajuste logger `uvicorn.access` no `main.py`.
- Se trocar modelos, verifique `VIDEO_CLASSES` e ajuste `VIDEO_THRESH_<NOME>` no `.env` para os novos labels.
- Ajustar `-hls_time` em `srt_reader.py` reduz frequência de polling (mais latência, menos requests).

---

Se quiser, eu transformo esse mapa num arquivo na pasta `docs/` com links e exemplos de comandos para cada área (DB, ML, Streams, Frontend).
