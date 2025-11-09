DOCKERIZATION GUIDE — Horus AI (backend)

## Purpose

This file documents an opinionated, production-ready approach to containerizing the Horus AI backend. It includes:

- A multi-stage Dockerfile for a minimal production image (Python 3.11-slim)
- Docker Compose (dev and production snippets) wiring Postgres, volumes and healthchecks
- Systemd service example to run the container on Linux hosts
- Healthcheck guidance (HTTP health endpoint) and uvicorn tuning
- Options for ffmpeg (installing in image vs mounting host binary)
- Notes on persistent volumes, permissions, non-root user, and security best-practices

## Use cases

- Local dev with Postgres: use docker-compose.dev
- Production single-host deployments: build image, run container with systemd or run directly via Docker + restart policy
- Kubernetes: use same health/readiness patterns and mount volumes for models and artifacts

## Quick PowerShell commands (dev)

# From repo root (PowerShell)

cd backend

# Build image locally

docker build -t horusai-backend:local .

# Run with a simple DB for quick smoke

docker run --rm -p 8000:8000 --name horusai-local -e DATABASE_URL=sqlite:///./dev.db horusai-backend:local

If you prefer compose (recommended for dev):
cd ..\
docker compose -f docker-compose.dev.yml up --build

## Multi-stage Dockerfile (recommended)

This Dockerfile produces a compact image and installs ffmpeg in the image (recommended for production where host may not provide ffmpeg). Adjust ffmpeg section if you want to rely on a host binary.

# (Place in backend/Dockerfile)

```dockerfile
# ---- build stage ----
FROM python:3.11-slim AS build
WORKDIR /app
# install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --upgrade pip && pip wheel --no-deps -r requirements.txt -w /wheels

# ---- runtime stage ----
FROM python:3.11-slim
LABEL maintainer="Horus AI"
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

# Install runtime deps and ffmpeg (smallest available in Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r horus && useradd --no-log-init -r -g horus horus
WORKDIR /app

# Copy wheels and install
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt || pip install -r requirements.txt

# Copy app code
COPY backend/app ./app
# Copy models & static if you want them baked in image (optional)
COPY backend/app/ml/models ./app/ml/models
COPY backend/static ./static

# Ensure data dirs exist and owned by non-root
RUN mkdir -p /app/static/clips /app/static/hls && chown -R horus:horus /app

USER horus
EXPOSE 8000

# Healthcheck: use a small endpoint that returns 200 (see docs for / or /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/ || exit 1

# Use uvicorn with sensible worker settings for CPU-bound TFLite work — tune for your instance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

## Notes on the Dockerfile

- We create a non-root user `horus` and chown the app directory to it for security.
- ffmpeg is installed in the image which simplifies deployment and avoids host binary mismatches.
- If you run many concurrent ffmpeg processes or need GPU acceleration, consider a different base image or using host ffmpeg.
- The `HEALTHCHECK` uses the root `/` (or `/health`) endpoint. Make sure your app exposes a lightweight health endpoint that returns 200 quickly.

## Docker Compose — dev example

Create `docker-compose.dev.yml` in repository root for local development (bind-mount code for fast iteration):

```yaml
version: "3.8"
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: horus
      POSTGRES_PASSWORD: horus
      POSTGRES_DB: horus
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "horus"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend/app:/app/app:delegated
      - ./backend/static:/app/static
      - ./backend/app/ml/models:/app/ml/models
    environment:
      DATABASE_URL: postgresql+psycopg2://horus:horus@db:5432/horus
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped

volumes:
  db_data:
```

## Compose — production notes

- For production, do NOT bind-mount code; build the image and reference the image tag in compose.
- Use named volumes for `static` and `models` if you want to upload new models at runtime.
- Set `restart: always` and appropriate resource limits.
- Secure secrets via environment variables from a secret store or Docker Secrets.

## Systemd unit (single-host production)

Example unit to run the container via systemd (create `/etc/systemd/system/horus-backend.service`):

```ini
[Unit]
Description=Horus AI backend container
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker run --rm --name horusai -p 8000:8000 \
  -e DATABASE_URL='postgresql+psycopg2://horus:horus@db:5432/horus' \
  -v /var/lib/horus/static:/app/static \
  -v /var/lib/horus/models:/app/ml/models \
  horusai-backend:latest
ExecStop=/usr/bin/docker stop horusai

[Install]
WantedBy=multi-user.target
```

Notes:

- Prefer running containers via a container manager (systemd + docker) or containerd runtime with proper restart/upgrade mechanism.
- Use a small wrapper script to stop/start containers during deployments and run DB migrations before starting.

## Healthchecks, probes and HTTP endpoints

- Implement a lightweight `/health` endpoint that checks:
  - DB connectivity (quick check)
  - Model files exist (optional)
  - Sufficient disk space for `static` folder (optional)
- Keep `/health` cheap — avoid loading models on each probe.
- Use Docker `HEALTHCHECK` and orchestrator liveness/readiness probes with `/health`.

## ffmpeg considerations

Three options to provide ffmpeg:

1. Install ffmpeg in the image (as in the Dockerfile above). Simpler for production; increases image size.
2. Mount host `ffmpeg` binary into the container (small image): `-v /usr/bin/ffmpeg:/usr/bin/ffmpeg:ro` but is OS dependent.
3. Run ffmpeg on the host and have backend just read the output folder (decoupled). This is robust for high-throughput setups.

If you install ffmpeg in the image, ensure you pick a distro-provided binary with compatible codecs. For Debian-based images:

```dockerfile
RUN apt-get update && apt-get install -y ffmpeg
```

## Persistent storage and permissions

- Mount host volumes for:
  - `/app/static` (HLS + clips) — ensure it is on a filesystem with enough I/O and retention policy
  - `/app/ml/models` (optional) — if you want to update models without rebuilding
- Set ownership to the non-root user used inside container (UID/GID mapping may be needed for host<->container). Example:

```bash
sudo chown -R 1000:1000 /var/lib/horus/static /var/lib/horus/models
```

- If using Docker on Linux, prefer using named volumes or a dedicated host path under `/var/lib/horus`.

## Kubernetes (skeleton)

A minimal Deployment snippet with liveness/readiness probes and volume mounts:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: horus-backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: horus-backend
  template:
    metadata:
      labels:
        app: horus-backend
    spec:
      containers:
        - name: backend
          image: your-registry/horusai-backend:latest
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 30
          volumeMounts:
            - name: static
              mountPath: /app/static
            - name: models
              mountPath: /app/ml/models
      volumes:
        - name: static
          persistentVolumeClaim:
            claimName: horus-static-pvc
        - name: models
          persistentVolumeClaim:
            claimName: horus-models-pvc
```

## CI / build pipeline (suggested)

1. Build and test
   - Install python deps, run `pytest` and linting
2. Build image
   - `docker build -t ghcr.io/<org>/horusai-backend:$GITHUB_SHA ./backend`
3. Push image to registry
4. Deploy (helm / kubectl apply / compose deploy) with new image tag

## Security & runtime best-practices

- Run as non-root inside container (we create `horus` user in Dockerfile).
- Do not bake secrets into image; pass via environment or secrets manager.
- Limit capabilities and use AppArmor/SELinux profiles where supported.
- Restrict `CORS` in production to your frontend domain.
- Protect admin endpoints (API key, IP whitelist, OAuth).

## Logging and rotation

- Prefer container stdout/stderr for logs; let the orchestrator collect them (e.g., Docker logging driver, Fluentd, or Elastic stack).
- For ffmpeg logs, write to `/app/static/hls/hls_ffmpeg.log` and configure logrotate on host if needed.

## Backup & retention

- Clips and HLS segments can consume significant disk; implement retention policies and automatic cleanup (cron job, or short TTL for HLS segments).

## Troubleshooting checklist

- Container won't start: `docker logs <container>` then `docker inspect` for healthcheck failures.
- 500s in browser and missing CORS header: check backend logs; a 500 might cause dev server to return index.html.
- `404` for clips: verify the file exists in the mounted `static/clips` folder and check container UID/GID permissions.

## Next steps I can help with

- Add an actual `backend/Dockerfile` and `docker-compose.yml` in the repo and test a local build with me.
- Add a lightweight `/health` endpoint to `main.py` if you don't already have one.
- Provide a Helm chart or k8s manifests with ConfigMap/Secret wiring for production deployments.

---

End of tools/DOCKER.md
