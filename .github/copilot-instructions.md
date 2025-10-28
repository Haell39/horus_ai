## Quick orientation for AI assistants

This repository is a two-part app: an Angular frontend (`/frontend`) and a FastAPI backend (`/backend/app`). Below are the minimal, high-value facts an automated coding agent needs to be productive here.

### Big picture

- Frontend: Angular app (see `frontend/package.json`, `frontend/angular.json`). Development: `ng serve` (script: `start`).
- Backend: FastAPI app mounted in `backend/app/main.py`. Routers live under `backend/app/api/endpoints/` (e.g. `ocorrencias`, `analysis`).
- Database: SQLAlchemy + PostgreSQL. Engine and session factory in `backend/app/db/base.py`; models in `backend/app/db/models.py`; Pydantic schemas in `backend/app/db/schemas.py`.
- ML: TFLite models live under `backend/app/ml/models/` and are loaded by `backend/app/ml/inference.py` (uses `tf.lite.Interpreter` via TensorFlow). `main.py` tries to call `inference.load_all_models()` on startup.

### Key files to reference when modifying behavior

- `backend/app/main.py` — app creation, CORS, router inclusion, optional startup model-loading.
- `backend/app/api/endpoints/analysis.py` — media upload → preprocess → TFLite inference → scheduling DB save (see `salvar_ocorrencia_task`). Use it as canonical example for file uploads, background tasks, and temporary-file handling.
- `backend/app/ml/inference.py` — model loading, preprocess*audio, preprocess_video_frame, and run*\*\_inference wrappers. If inference fails, check `models_loaded`, interpreter objects and file paths `AUDIO_MODEL_PATH`/`VIDEO_MODEL_PATH`.
- `backend/app/db/base.py` — DB engine and FastAPI dependency `get_db()`; any endpoint needing DB should depend on `get_db`.

### Developer workflows (practical commands)

- Backend (dev): create a Python venv, install `backend/requirements.txt`, provide a `.env` with `DATABASE_URL`, then run:

```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Frontend (dev):

```powershell
cd frontend
npm install
npm start   # runs `ng serve` per package.json
```

Notes: `moviepy` and some ML packages require system binaries (ffmpeg) and correct TensorFlow build. `requirements.txt` lists `tensorflow` — ensure matching platform wheel (CPU vs GPU).

### Project-specific conventions & patterns

- Routers are grouped under `backend/app/api/endpoints/`. To add an endpoint: create a new router module there and include it in `main.py` with `app.include_router(..., prefix='/api/v1')`.
- DB transactions: endpoints receive DB sessions via `Depends(get_db)` from `backend/app/db/base.py`. Use `db.add()`, `db.commit()`, and `db.refresh()` as in `analysis.py`.
- Background work: long-running persistence is delegated to FastAPI `BackgroundTasks` (see `background_tasks.add_task(salvar_ocorrencia_task, db, ...)`). If modifying, ensure DB sessions are safe to use from background tasks (copy data, avoid long-lived session objects).
- ML behavior: `inference.load_all_models()` is called at import time and attempted at startup. The code checks `inference.models_loaded` before inference and returns a structured AnalysisOutput instead of throwing, e.g. when models are missing.
- Logging: code uses `print()` statements for informational and debug logs. Follow existing style (prints rather than structured logging) unless you refactor logging consistently project-wide.

### Integration points and gotchas

- Model files: `backend/app/ml/models/*.tflite`. If absent, `inference.models_loaded` will be False. Check file names in `inference.py` (`audio_model_quant.tflite`, `video_model_quant.tflite`).
- Config: `backend/app/core/config.py` loads `.env` and exposes `settings.DATABASE_URL`. Ensure `.env` is present in repository root for local dev.
- CORS: `main.py` currently uses `allow_origins=['*']`. In production restrict this to the frontend host.
- Temporary files: `analysis.py` writes uploads to `tempfile.NamedTemporaryFile(delete=False, ...)` and removes them in a `finally` block — preserve this safety pattern when reusing uploads.

### Quick troubleshooting checks (where to look first)

- If DB errors → inspect `backend/app/core/config.py` and your `.env` `DATABASE_URL`, then `backend/app/db/base.py` for engine creation.
- If ML errors → confirm TFLite files exist, then check `backend/app/ml/inference.py` logs and `models_loaded` at startup.
- If uploads fail → check `analysis.py` for file saving, MoviePy usage (ffmpeg), and file permissions.

If anything here is unclear or you want a different focus (e.g., more frontend guidance, testing, or CI), tell me which area to expand and I'll update this file.
