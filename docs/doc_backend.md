# Horus AI — Backend Reference

This document describes the `backend` part of the Horus AI project in detail.
It is intended for two audiences:

- An automated agent / AI that needs a precise map of the API surface, data shapes, and runtime behavior.
- A human engineer (Dev / SRE / ML engineer) who will maintain, extend, dockerize, or deploy the backend.

Keep this file in sync with code changes. If you add endpoints, change environment variables, or move files, update this doc.

---

## Table of contents

1. Overview
2. Repo layout (backend)
3. Runtime & requirements
4. Configuration & environment
5. How to run (dev) — PowerShell
6. API surface (endpoints summary)
7. Data models and schemas
8. Storage and filesystem layout
9. Streams, HLS and ffmpeg integration
10. ML models (TFLite) and inference
11. Adding a new model or upgrading models
12. Background processing and jobs
13. WebSocket events
14. Admin operations and maintenance
15. Dockerization / containerization notes
16. Observability, logs and troubleshooting
17. Testing strategy
18. Security considerations
19. CI / CD and recommended workflow
20. FAQs and common operations

---

## 1 — Overview

The backend is a FastAPI application responsible for:

- Accepting SRT streams and creating HLS output (via ffmpeg subprocesses).
- Extracting frames and running lightweight TFLite inference on video frames (and audio segments).
- Persisting detected occurrences (events) into a relational database via SQLAlchemy.
- Serving static artifacts: HLS playlists/segments and generated clips under `/hls` and `/clips`.
- Exposing REST endpoints to control ingest, upload files for offline analysis, list occurrences, and admin utilities.
- Broadcasting new occurrences in realtime over a WebSocket endpoint `/ws/ocorrencias`.

Design goals: small, debuggable, and portable; the ML inference is intentionally lightweight (TFLite) to allow local CPU inference.

## 2 — Repo layout (backend)

Key files and directories (relative to repository root):

- `backend/app/main.py` — application bootstrap. mounts static folders (`/clips`, `/hls`), registers routers, sets up CORS and startup logic to load ML models.
- `backend/app/api/endpoints/` — FastAPI routers (per-area):
  - `ocorrencias.py` — list/create occurrences (HTTP GET/POST)
  - `analysis.py` — upload & analysis endpoints, inline & background processing
  - `streams.py` — start/stop/status for SRT ingest
  - `ws.py` — websocket router for broadcasting occurrences
  - `admin.py` — admin helpers (disk usage, storage config, sync)
  - `docs.py` — (added) listing and serving repository markdown for the frontend docs panel
- `backend/app/core/` — configuration helpers and storage helpers
  - `config.py` — typed settings (environment variables)
  - `storage.py` — small storage config reader/writer and `get_clips_dir()` helper
- `backend/app/db/` — DB models, schemas and base session
  - `base.py` — SQLAlchemy engine / SessionLocal and Base
  - `models.py` — ORM `Ocorrencia` model
  - `schemas.py` — Pydantic schemas for API input/output
- `backend/app/ml/` — ML inference wrappers and TFLite interpreter loading
  - `inference.py` — wrappers `analyze_video_frames`, `analyze_audio_segments`, constants for model filenames
  - `models/*.tflite` — TFLite model files (committed small quantized models)
- `backend/static/` — files served by StaticFiles
  - `clips/` — generated clips (served at `/clips/...`)
  - `hls/` — playlist + .ts segments (served at `/hls/...`)
- `backend/.env.example` — example env variables
- `backend/requirements.txt` — python dependencies

## 3 — Runtime & requirements

- Python 3.11+
- Recommended venv for local dev (PowerShell examples below).
- Dependencies: see `backend/requirements.txt` (FastAPI, SQLAlchemy, uvicorn, tensorflow, moviepy, etc.).
- ffmpeg installed and available on PATH (required for HLS/segmenting/cutting clips).
- A running Postgres instance is recommended for persistence; the code can run with SQLite for quick tests but production uses Postgres configured via `DATABASE_URL`.

## 4 — Configuration & environment

Main configuration is in `backend/app/core/config.py` and environment variables. The important variables:

- `DATABASE_URL` — SQLAlchemy database URI (ex: `postgresql+psycopg2://user:pass@host:5432/dbname`).
- `VIDEO_VOTE_K`, `VIDEO_MOVING_AVG_M`, `VIDEO_THRESH_DEFAULT` — inference tuning parameters (default exposed in startup logs).
- `VIDEO_THRESH_<CLASS>` — per-class thresholds optionally configured.
- For local dev, copy `.env.example` to `.env` and adjust values.

## 5 — How to run (dev) — PowerShell

Recommended steps (PowerShell):

1. Backend

```powershell
cd backend
# create and activate venv (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# run without --reload when testing streams (avoids killing ffmpeg subprocesses)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Frontend

```powershell
cd frontend
npm install
npm start
```

Notes:

- If you make code changes to backend Python modules, restart the backend process.
- Avoid `--reload` while testing stream ingestion (ffmpeg subprocesses will be terminated on reload).

## 6 — API surface (endpoints summary)

All endpoints are mounted under `/api/v1` (unless noted). This is a quick reference; consult the router source files for full details and request/response shapes.

- GET `/api/v1/ocorrencias` — list occurrences (pagination optional)
- POST `/api/v1/ocorrencias` — create occurrence (used by internal code and tests)
- POST `/api/v1/analysis/upload` — upload a video file for analysis (multipart/form-data). Small files may be processed inline and create an occurrence; large files are queued.
- POST `/api/v1/analyze` — endpoint for analyzing a single media file (audio/video) (used by frontend or integrations)
- POST `/api/v1/streams/start` — start ingest SRT → HLS (body: { url, fps? })
- POST `/api/v1/streams/stop` — stop ingest
- GET `/api/v1/streams/status` — check ingest status
- WebSocket `/ws/ocorrencias` — clients can subscribe to realtime occurrence notifications
- Admin:
  - GET `/api/v1/admin/disk-usage?path=...` — returns disk usage stats for a path
  - GET `/api/v1/admin/storage-config` — read storage configuration
  - POST `/api/v1/admin/storage-config` — update storage configuration
  - POST `/api/v1/admin/sync-clips` — copies files from configured storage dir into public `backend/static/clips`
- Docs viewer (internal frontend feature):
  - GET `/api/v1/docs/list` — lists README/MD files in repo `docs`, `tools`, `configs`
  - GET `/api/v1/docs/file?folder=docs&name=README.md` — returns markdown content (read-only)

Note: Some endpoints are intended for admin/dev only; secure them in production.

## 7 — Data models and schemas

Primary persisted model: `Ocorrencia` (occurrence/event)

Fields (important ones):

- `id` (int, PK)
- `start_ts` (timestamp with timezone) — UTC-aware start of event
- `end_ts` (timestamp with timezone) — UTC-aware end of event
- `duration_s` (float)
- `category` (string) — e.g. 'video-file', 'Video Arquivo'
- `type` (string) — predicted failure class (e.g. 'bloco', 'borrado')
- `severity` (string) — 'Leve (C)', 'Média (B)', 'Grave (A)', 'Gravíssima (X)'
- `confidence` (float) — model confidence score
- `evidence` (JSON) — metadata: clip_path, clip_duration_s, original_filename, event_window, model

Pydantic schemas (in `db/schemas.py`) provide `OcorrenciaCreate`, `OcorrenciaRead` used by endpoints and websocket payloads.

Important contract for agents: when the API returns occurrences, timestamps are ISO strings with UTC offset (e.g. `2025-11-08T22:27:25.824969+00:00`). Frontend parses these and displays in local timezone.

## 8 — Storage and filesystem layout

- `backend/static/clips` — public folder mounted at `/clips` by FastAPI. Any file added here is accessible at `http://<host>:8000/clips/<name>`.
- `backend/static/hls` — HLS output (`stream.m3u8` + `.ts` segments) mounted at `/hls`.
- Configurable storage: `app.core.storage` reads/writes `.storage_config.json` (root) to allow choosing an external folder (e.g. `D:\Videos`).
  - When using an external folder, the code attempts to copy generated clips into `backend/static/clips` so they are served. If you prefer to serve external folder directly, update `main.py` to mount it or set storage config accordingly.

## 9 — Streams, HLS and ffmpeg integration

Flow:

1. `streams.SRTIngestor` starts ffmpeg subprocess to read SRT remote and generate HLS into `backend/static/hls`.
2. A separate ffmpeg process or a watcher extracts short clips/frames for inference.
3. The player in the frontend requests `/hls/stream.m3u8` and plays segments.

Notes and common issues:

- ffmpeg must be in PATH. If `stream.m3u8` or `.ts` segments are not appearing, check `backend/static/hls/hls_ffmpeg.log`.
- Don't use `--reload` for uvicorn when running ingest; reload kills subprocesses.
- If ffmpeg logs show codec errors (`Missing reference picture`, `sps_id 0` etc.), the remote SRT stream may be incompatible or contain corrupted NALs.

Useful commands (PowerShell):

```powershell
Get-Content backend\static\hls\hls_ffmpeg.log -Tail 200
Get-ChildItem backend\static\hls\
Get-Process -Name ffmpeg | Select-Object Id,ProcessName,StartTime
```

To forcibly kill lingering ffmpeg processes:

```powershell
Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

## 10 — ML models (TFLite) and inference

Location: `backend/app/ml/` and `backend/app/ml/models/`

Files:

- `video_model_quant.tflite`, `audio_model_quant.tflite` — quantized TFLite models checked into the repo for convenience (small / sample models).
- `inference.py` exports helper functions used by endpoints and background worker:
  - `analyze_video_frames(path) -> (pred_class, confidence, event_time)` — runs frame-level inference and returns predicted class, confidence and timestamp (seconds) of detected event relative to clip start (or None).
  - `analyze_audio_segments(path) -> (pred_class, confidence)` — audio-based inference.

Runtime notes:

- In `main.py` on startup the code loads TFLite interpreters and XNNPACK delegate where available.
- The inference code is optimized for local CPU using quantized models.

## 11 — Adding or upgrading models

Steps to add a new TFLite model (video or audio):

1. Train/export model and convert to TFLite with quantization (recommended for CPU inference).
2. Place model file in `backend/app/ml/models/` and update constant names in `inference.py` (e.g. `VIDEO_MODEL_FILENAME`).
3. Update model input preprocessing and postprocessing code in `inference.py` (shape, normalization, argmax mapping to classes).
4. Update `core/config.py` and any `VIDEO_THRESH_<CLASS>` env vars if new classes require thresholds.
5. Restart backend; models are loaded on startup.

Testing locally:

```powershell
# run a small offline analysis on a file
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/analysis/upload -InFile .\tests\sample_video.mp4 -ContentType 'multipart/form-data'
```

Tips:

- Keep model names and a small metadata mapping (class->human label) next to the model file to avoid mismatches.
- Add unit tests for the new preprocessing logic.

## 12 — Background processing and jobs

- For larger uploads, the endpoint enqueues background tasks using FastAPI's `BackgroundTasks` (synchronous function executed after response) or internal worker function `SessionLocal` for DB access.
- When events are detected in background worker, they are persisted and then broadcast to WebSocket clients.

Considerations for scaling:

- For high-throughput or production workloads, move background jobs to a separate worker process (Celery / Redis, RQ) so that long-running operations do not block the web process.
- Ensure the worker has access to the same models and storage paths.

## 13 — WebSocket events

- WebSocket endpoint: `/ws/ocorrencias` (manager in `app.websocket_manager`).
- When an occurrence is saved, code constructs `OcorrenciaRead` and broadcasts JSON payload `{ type: 'nova_ocorrencia', data: <OcorrenciaRead dict> }` to all connected clients.

Clients (frontend) should expect the `data` field to contain ISO timestamps and `evidence.clip_path` (public path under `/clips`).

## 14 — Admin operations and maintenance

Available admin endpoints (dev/ops):

- `/api/v1/admin/disk-usage?path=...` — check disk usage for storage monitoring UI.
- `/api/v1/admin/storage-config` GET/POST — read/write storage configuration (local path vs oneDrive link). In production protect these endpoints.
- `/api/v1/admin/sync-clips` POST — copies files from storage dir into `backend/static/clips` so they become served by FastAPI. Useful when clips are generated in an external folder.

File sync and permission notes:

- The backend will attempt to copy generated clips into `backend/static/clips`. If copy fails due to permissions or cross-volume issues, check backend logs which now include diagnostic messages.

## 15 — Dockerization / containerization notes

Minimal Dockerfile example (backend) — suitable as a starting point:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
# copy models and static if you want baked into image (optional)
COPY backend/app/ml/models ./app/ml/models
COPY backend/static ./static
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Docker Compose (dev) example to run backend + Postgres and map host ffmpeg (optional):

```yaml
version: '3.8'
services:
	db:
		image: postgres:15
		environment:
			POSTGRES_USER: horus
			POSTGRES_PASSWORD: horus
			POSTGRES_DB: horus
		volumes:
			- db_data:/var/lib/postgresql/data

	backend:
		build: ./backend
		depends_on: [db]
		environment:
			DATABASE_URL: postgresql+psycopg2://horus:horus@db:5432/horus
		ports:
			- '8000:8000'
		volumes:
			- ./backend/static:/app/static
			- ./backend/app/ml/models:/app/app/ml/models
		devices: [] # if you plan to expose hardware devices, or mount ffmpeg from host

volumes:
	db_data:
```

Notes:

- ffmpeg is usually installed on the host. If you need ffmpeg inside the container, install system packages (apt) in the Dockerfile and include ffmpeg binary.
- If you mount external storage (host path) into container, ensure permissions allow container user to read/write.

## 16 — Observability, logs and troubleshooting

Where to look:

- Backend stdout/stderr — uvicorn process logs (startup, route registrations, model loading).
- `backend/static/hls/hls_ffmpeg.log` — ffmpeg logs for HLS creation.
- Model load logs — printed at startup by `inference` loader (see `main.py` startup event).
- DB errors — check uvicorn logs and the database server logs.

Common symptoms and actions:

- `404` for `/clips/<file>`: file not present in `backend/static/clips`. Use `/api/v1/admin/sync-clips` to copy from storage dir, or ensure the code successfully copied the file and that file permissions allow read.
- `No 'Access-Control-Allow-Origin' header` in browser: often caused by backend returning 500 and dev-server serving HTML. Fix backend exception or configure CORS in `main.py` (by default dev config allows `*`).
- `Unexpected token '<'` when parsing JSON in frontend: indicates frontend called dev server instead of backend; configure absolute backend URL in dev or proxy dev server.

## 17 — Testing strategy

Unit and integration testing recommendations:

- Unit tests for `inference.py` preprocessing functions and postprocessing.
- Integration tests for `analysis.upload` using small synthetic clips — exercise inline and background branches.
- Mock TFLite interpreter in unit tests to validate control flow without heavy libs.

Example using pytest (pseudo):

```python
def test_calcular_severidade(tmp_path):
		# create tiny mp4 or mock duration and call calcular_severidade_e_duracao
		assert calcular_severidade_e_duracao(...) == (expected_duration, expected_severity)
```

End-to-end: run uvicorn (in test env) and exercise endpoints with small files using `requests` or `Invoke-RestMethod`.

## 18 — Security considerations

- Admin endpoints (`/api/v1/admin/*`) are currently unprotected; secure them before exposing the API in production (API key, OAuth, or internal network only).
- CORS: currently permissive for local dev. In production, set `allow_origins` to the frontend origin.
- Avoid committing secrets to repo; use environment variables or Vault for model credentials or external storage credentials.
- Sanitize any user-generated content served as HTML (we return Markdown content as raw text via JSON; rendering is done on frontend with sanitation).

## 19 — CI / CD and recommended workflow

Suggested pipeline steps:

1. Static checks: flake8 / pylint, mypy (if types are used), black formatting.
2. Install requirements and run unit tests (pytest).
3. Build frontend (npm ci && npm run build) to ensure no regressions.
4. Build container image (docker build) and run smoke tests against container.
5. Publish images and deploy.

Use GitHub Actions / Azure Pipelines / GitLab CI to implement above.

## 20 — FAQs and common operations

- Q: Where are clip files stored?

  - A: By default in `backend/static/clips`. You can configure `storage_core` to use an external folder; use admin sync to copy files into the public folder.

- Q: Why is my time shifted in UI?

  - A: Backend stores UTC-aware timestamps. Frontend converts to client local timezone. If frontend receives a naive ISO string, it assumes UTC. See `frontend/src/app/pages/cortes/cortes.component.ts` for parsing strategy.

- Q: How to add a new model?
  - A: See section "Adding or upgrading models" above — add TFLite to `app/ml/models` and update `inference.py`.

---

If you'd like, I can also generate a `backend/DOCKER.md` with an opinionated production-ready container spec (systemd, healthchecks, ffmpeg packaging), or add a `docs/` entry that documents how to add a new class to the model and update thresholds end-to-end.
