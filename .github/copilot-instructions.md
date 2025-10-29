## Quick orientation for AI assistants and contributors

This repository is a two-part application used for ingesting live SRT streams, running lightweight ML inference (TFLite), persisting detections, and exposing a web UI for realtime monitoring. It contains:

- Frontend: Angular app at `frontend/` (UI & player).
- Backend: FastAPI app at `backend/app/` (API, streaming glue, ML inference, DB access).
- Static output: `backend/static/` (HLS playlist + segments in `hls/`, generated clips in `clips/`).

This document summarizes the architecture, developer workflows, important files, streaming flow (SRT → HLS → player), troubleshooting tips and recommended next steps. Keep it up to date as you change behavior in the project.

---

## 1 — Big picture (what each piece does)

- Frontend (`frontend/`)

  - Angular single-page app. The Monitoramento page attaches to the HLS playlist served by the backend and listens to a WebSocket for new occurrences.
  - Uses hls.js (CDN) when the browser doesn't support native HLS. UI includes Start/Stop controls that call backend stream endpoints (SRT → HLS).

- Backend (`backend/app/`)

  - FastAPI service. Routers live under `backend/app/api/endpoints/` (examples: `ocorrencias`, `analysis`, `streams`, `ws`).
  - Provides REST endpoints for analysis, occurrences CRUD and stream control. Serves static HLS/Clips via `StaticFiles` mounts.
  - ML inference uses TFLite via `backend/app/ml/inference.py` and loads model files from `backend/app/ml/models/`.

- Database

  - SQLAlchemy models in `backend/app/db/models.py` and session factory in `backend/app/db/base.py`.
  - Use dependency `get_db()` in endpoints for safe sessions.

- Streaming pipeline (SRT → HLS)
  - The backend controls `ffmpeg` processes to pull an SRT stream and remux into HLS segments written to `backend/static/hls/`.
  - A separate ffmpeg process extracts frames (image sequence) for per-frame ML inference and optional clip creation.
  - The backend exposes control endpoints to start/stop streams and implements safe cleanup of transient files.

---

## 2 — Key files and responsibilities (quick map)

- `backend/app/main.py` — app creation, CORS, router inclusion, StaticFiles mounts (`/hls`, `/clips`), and optional startup ML loading.
- `backend/app/api/endpoints/streams.py` — Start/Stop/Status endpoints for stream ingest.
- `backend/app/streams/srt_reader.py` — SRTIngestor: manages ffmpeg processes (HLS + frame extractor), per-frame watch loop, persistence of occurrences, and cleanup logic.
- `backend/app/api/endpoints/analysis.py` — file upload → preprocess → ML inference → background persistence; canonical example for uploads and BackgroundTasks.
- `backend/app/api/endpoints/ws.py` — WebSocket ConnectionManager and broadcast helper used to send real-time occurrences to frontend.
- `backend/app/ml/inference.py` — model loading and wrapper functions (audio/video preprocess + run inference). Checks `models_loaded` flag.
- `backend/app/db/*` — SQLAlchemy models, schemas and DB session helper.
- `frontend/src/app/pages/monitoramento/*` — Monitoramento component (player, start/stop, reconnection, alert list).

---

## 3 — How to run (developer / local) — Windows PowerShell

Prereqs:

- Python 3.10+ (matching `requirements.txt`), Node.js + npm, ffmpeg installed and in PATH.
- `.env` file at repo root with `DATABASE_URL` (Postgres) if you want DB persistence.

Backend (recommended for testing streaming reliably):

```powershell
cd backend
# create & activate venv (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# run without reload while testing streams to avoid worker restarts
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm start    # runs ng serve (dev server) per package.json
```

Notes: avoid `--reload` on the backend while stress-testing streaming — automatic worker restarts will close WebSockets and ffmpeg processes.

---

## 4 — Stream control & endpoints (what to call)

- POST /api/v1/streams/start { url, fps? }

  - Starts SRT → HLS ingest for the provided `srt://` URL.
  - Returns success only when the HLS playlist is ready (client UX improved to attach after Start).

- POST /api/v1/streams/stop

  - Stops ingest processes and cleans `backend/static/hls/` (writes a minimal playlist with #EXT-X-ENDLIST).

- GET /api/v1/streams/status

  - Returns {'running': bool, 'fps': number}

- WebSocket: /ws/ocorrencias

  - Broadcasts new occurrences (id, timestamps, type, confidence, evidence) to connected frontend clients.

- Static HLS: GET /hls/stream.m3u8 and segments under /hls/\*.ts
- Clips: GET /clips/\*.mp4

---

## 5 — Streaming reliability notes & troubleshooting

- If ffmpeg prints errors like `Connection to srt://... failed: I/O error` or SRT logs `ERROR:BACKLOG`, the remote SRT endpoint rejected the handshake (listener backlog). This is an upstream issue — either retry/connect later or coordinate with the SRT server operator.
- H.264 warnings in ffmpeg logs (sps_id out of range, non-existing PPS, Missing reference picture) indicate corrupted or out-of-order NALs. We added tolerant ffmpeg flags in the HLS command to reduce impact:
  - `-fflags +genpts+igndts -avoid_negative_ts make_zero -use_wallclock_as_timestamps 1` helps create stable timestamps.
- If browser shows 404 for segments after Stop or during start, check the playlist: `backend/static/hls/stream.m3u8` and the ffmpeg log `backend/static/hls/hls_ffmpeg.log`.
- To inspect HLS runtime (PowerShell):

```powershell
Get-ChildItem backend\static\hls\
Get-Content backend\static\hls\hls_ffmpeg.log -Tail 200
Get-Content backend\static\hls\stream.m3u8 -Raw
```

- Kill stray ffmpeg processes (if necessary):

```powershell
Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 6 — Logs, cleanup & git

- HLS logs and segments are written to `backend/static/hls/`. This folder is included in `.gitignore` to avoid committing transient media.
- `stop()` in the ingestor cleans the folder and writes a minimal playlist with `#EXT-X-ENDLIST` to signal clients that streaming ended.

---

## 7 — ML models and inference

- TFLite models expected under `backend/app/ml/models/` (e.g. `audio_model_quant.tflite`, `video_model_quant.tflite`).
- `backend/app/ml/inference.py` exposes helpers: `load_all_models()`, `run_audio_inference`, `run_video_inference`, etc. On startup `main.py` calls `load_all_models()` and code checks `models_loaded`.

---

## 8 — Developer testing & quick checks

- Health: GET `/` returns app message.
- Streams status: GET `/api/v1/streams/status`.
- Try start (PowerShell):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body (@{url='srt://...'; fps=1.0} | ConvertTo-Json) -ContentType 'application/json'
```

- Stop: POST `/api/v1/streams/stop` (no body).

---

## 9 — Recommended next steps & improvements

- Move SRT connection secrets (passphrases) out of the frontend and into a secure backend `.env` or vault; expose only identifiers/controls via API.
- Add retry/backoff in `SRTIngestor.start()` to automatically try a few reconnects when the remote SRT rejects briefly (we already added better checks; more retries help flaky upstream servers).
- Add a dedicated `/api/v1/streams/cleanup` admin endpoint to kill leftover ffmpeg processes and wipe HLS for easier recovery.
- Add structured logging (e.g., `logging` module) and optional file rotation for `hls_ffmpeg.log` to reduce growth.
- Add CI checks and a lightweight integration test that simulates a short SRT stream (or mock) and verifies that `/streams/start` → playlist appears.

---

If you want, I can also add a compact `README.md` (top-level) with a developer Quick Start that re-uses these commands and highlights the streaming troubleshooting checklist.

If you'd like sections tailored for non-dev stakeholders (DevOps/Infra, QA, Product), tell me which audience to prioritize and I will add short runbooks.
