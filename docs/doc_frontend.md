
# Horus AI — Frontend Reference

This document describes the `frontend` part of the Horus AI project. It's written for two audiences:

- A developer (frontend engineer, integrator) who will extend, debug or productionize the Angular app.
- An automated agent that needs a precise map of pages, services, APIs used, and build/deploy steps.

Keep this document up to date when moving components, changing routes, or altering API contracts.

---

## Table of contents

1. Overview
2. Repo layout (frontend)
3. Runtime & local requirements
4. How to run (dev) — PowerShell
5. Configuration & environment
6. Key pages and components
7. Services and API contracts
8. Models / TypeScript interfaces
9. Playback: HLS player & WebSocket
10. Docs viewer (new feature) — quick notes
11. Building for production
12. Tests and linting
13. CI / CD recommendations
14. Troubleshooting common issues
15. Adding features & conventions
16. Performance & bundle-size tips
17. Next steps & recommended improvements

---

## 1 — Overview

The frontend is an Angular application (dev server on port 4200 by default). It provides:

- Live monitoring page that plays HLS served by the backend.
- A clips/cortes page listing generated clips and opening/download links.
- Configurações page with several cards, including the Documentation card that lists `.md` files from repository folders.
- Services that call backend endpoints under `http://localhost:8000/api/v1` in development.

Design goals: small, modular, and easy to run locally; keep business logic in services and lightweight presentational components.

## 2 — Repo layout (frontend)

Key files and directories (relative to repo root):

- `frontend/angular.json`, `frontend/package.json` — build config and dependencies.
- `frontend/src/main.ts` — app bootstrap.
- `frontend/src/index.html` — app HTML shell.
- `frontend/src/app/` — app code:
	- `app.component.ts` — root component and common layout
	- `app.routes.ts` — routing configuration
	- `app.config.ts` — frontend-side configuration constants
	- `pages/` — feature pages (monitoramento, cortes, dados, cortes, configuracoes)
	- `services/` — HTTP services, WebSocket helpers, and small utilities (e.g. `ocorrencia.service.ts`, `docs.service.ts`)
	- `models/` — TypeScript models (e.g. `ocorrencia.ts`)
	- `shared/` — common components (sidebar, preloader)
- `frontend/src/environments/` — environment.ts (dev/prod to inject backend base URL)

## 3 — Runtime & local requirements

- Node.js (recommended 18.x or later) and npm.
- Angular CLI (dev) — installed as project devDependency; use `npm start` to run dev server.

Install & run (PowerShell):

```powershell
cd frontend
npm install
npm start
```

Notes:
- The dev server runs on `http://localhost:4200/`.
- In development the `docs.service.ts` uses `http://localhost:8000` as the backend host to avoid dev-server proxy issues. For production builds, environment files should point to the real backend.

## 4 — How to run (dev) — PowerShell

Steps for a typical dev session:

1. Start backend (see backend docs)
2. Start frontend

```powershell
cd frontend
npm install
npm start
```

3. Open `http://localhost:4200` in your browser.

If you change backend host or port during dev, edit `frontend/src/environments/environment.ts` or switch the service to call a different base URL.

## 5 — Configuration & environment

Frontend configuration lives in `src/environments`:

- `environment.ts` — development values (used by `ng serve`)
- `environment.prod.ts` — production values (used by `ng build --prod`)

Important values to check:

- `apiBaseUrl` — base URL for backend API calls. In dev it may be hard-coded to `http://localhost:8000` to avoid dev-server proxy issues.

For production, ensure `environment.prod.ts` points to the deployed backend and that CORS is properly configured on backend.

## 6 — Key pages and components

High-level mapping (path -> purpose):

- `src/app/pages/monitoramento/` — Monitoring UI, HLS player + WebSocket events listing occurrences in realtime.
- `src/app/pages/cortes/` — Clips listing, with controls to download or play clips served under `/clips`.
- `src/app/pages/configuracoes/` — Config page, hosts cards including the new Documentation card component.
- `src/app/components/gerenciar-clipes/` — UI for managing clips (delete, copy to storage, reprocess).
- `src/app/shared/preloader/` & `sidebar/` — shared UI controls used across pages.

Component development notes:

- Prefer smaller presentational components; keep HTTP logic inside services.
- Use Angular's `ChangeDetectionStrategy.OnPush` for high-frequency UI updates (e.g., monitoring table) if you need performance improvements.

## 7 — Services and API contracts

Services live under `src/app/services/` and encapsulate API calls and WebSocket connections.

Important services:

- `ocorrencia.service.ts` — CRUD for occurrences, used by monitoring and cortes pages.
	- Expected endpoints: GET `/api/v1/ocorrencias`, POST `/api/v1/ocorrencias`.
- `docs.service.ts` — lists files and retrieves markdown content for the Documentation card.
	- GET `/api/v1/docs/list`
	- GET `/api/v1/docs/file?folder=...&name=...`
- `preloader.service.ts` / `tema.service.ts` — UI helpers for loading state and theme selection.

HTTP contract notes for agents:

- Occurrence objects contain ISO timestamps with timezone (UTC). Frontend converts to local timezone for display.
- `docs.list()` returns a JSON object with arrays for `docs`, `tools`, `configs`, each entry containing `name`, `size`, and `mtime` (mtime is an ISO timestamp).

## 8 — Models / TypeScript interfaces

Key TS interfaces live under `src/app/models`.

Example `Ocorrencia` (simplified):

```ts
export interface Ocorrencia {
	id: number;
	start_ts: string; // ISO timestamp with timezone
	end_ts?: string;
	duration_s?: number;
	category?: string;
	type?: string;
	severity?: string;
	confidence?: number;
	evidence?: any; // expand to a typed interface if needed
}
```

Keep these interfaces in sync with backend Pydantic schemas; consider generating TypeScript types from an OpenAPI spec in the future.

## 9 — Playback: HLS player & WebSocket

HLS playback

- The player points to `http://<backend>:8000/hls/stream.m3u8`.
- Use a modern player (e.g., hls.js for browsers that don't support native HLS) — the frontend already includes a player wrapper.

WebSocket

- The frontend subscribes to occurrences using `ws://<backend>:8000/ws/ocorrencias`.
- The WS payload format: `{ type: 'nova_ocorrencia', data: Ocorrencia }`.
- Implement reconnection/backoff logic in the WebSocket manager to handle backend restarts.

## 10 — Docs viewer (new feature) — quick notes

- The Documentation card uses `docs.service.ts` to list MD files and fetch contents.
- In dev the service calls `http://localhost:8000` explicitly to avoid dev-server returning `index.html` for non-proxied API calls.
- The card renders sanitized HTML produced from a small markdown renderer in the component. For richer MD features, consider adding a full markdown library (marked, ngx-markdown) and mounting static assets.

## 11 — Building for production

Build command:

```powershell
cd frontend
npm ci
npm run build -- --configuration production
```

This will create a `dist/` folder with static assets. You can serve this folder using any static file server or serve it from the backend via `StaticFiles` mount.

Important:

- Ensure `environment.prod.ts` contains the correct `apiBaseUrl`.
- Address any `CommonJS` warnings produced by Angular — some libraries warn when not ESM.
- Production builds are sensitive to bundle size. See section 16 for tips.

## 12 — Tests and linting

Project includes unit tests and component specs (Karma/Jasmine or Jest depending on config).

Commands:

```powershell
cd frontend
npm test       # run unit tests (dev)
npm run lint   # run tslint/eslint checks
```

Recommendations:

- Add a few integration tests for the `DocumentationCard` and `Cortes` page to validate end-to-end interactions (mock backend responses).
- Use `ng test --watch=false` in CI to run tests once.

## 13 — CI / CD recommendations

Suggested pipeline:

1. Install dependencies (`npm ci`) and run lint/tests.
2. Build production bundle (`npm run build -- --configuration production`).
3. Archive `dist/` and optionally copy artifacts to a static hosting service (S3, Netlify) or copy into backend `static` folder for serving.

Optional: generate an OpenAPI client for TypeScript to keep API contracts in sync.

## 14 — Troubleshooting common issues

- Unexpected token '<' when hitting backend JSON endpoint: dev-server is returning `index.html` because the request went to the frontend dev server instead of the backend. Fix by using absolute backend URL (e.g., `http://localhost:8000`) or configure the Angular dev server proxy.
- CORS errors: backend must include the frontend origin or `*` in `CORs` settings during testing.
- 404 on clip link: the clip file is not present in backend `static/clips` or path mismatch between frontend stored `evidence.clip_path` and public URL.
- WebSocket disconnects: verify backend is running and that the URL uses `ws://` or `wss://` correctly.

## 15 — Adding features & conventions

Conventions used in this repo:

- Services handle HTTP and data transformations; components are presentational.
- Prefer `async`/`await` with `HttpClient` RxJS conversions inside services for clarity.
- Keep styles scoped to components; use a small shared theme service for colors.

When adding a new page/component:

1. Add route entry in `app.routes.ts`.
2. Create page under `pages/<your-page>` and expose a small, testable public API for the component.
3. Add service method(s) to call backend endpoints; keep API surface minimal and typed.
4. Add unit tests for component(s) and service(s).

## 16 — Performance & bundle-size tips

- Use `ChangeDetectionStrategy.OnPush` on components that don't need frequent full-tree checks.
- Lazy-load feature modules in routes to reduce initial bundle.
- Prefer ESM-compatible libraries. The Angular build warns when CommonJS-only packages are used; consider replacing them.
- Compress images and serve them via CDN or `static/` folder.

## 17 — Next steps & recommended improvements

- Add e2e tests (Cypress or Playwright) that exercise main user flows: start backend, open player, receive a fake WS event, play clip.
- Replace the small inline markdown renderer with `ngx-markdown` + `marked` for better fidelity and asset support.
- Add TypeScript types auto-generated from the backend OpenAPI spec.
- Add a `frontend/README.md` with quick dev notes and common Gotchas for new contributors.

---

If you'd like, I can now:
- Generate `frontend/README.md` summarizing these instructions as a quick-start for new contributors, or
- Add an Angular dev-server proxy config (`proxy.conf.json`) and update `package.json` to use it so calls to `/api` are proxied to `http://localhost:8000` in dev, or
- Add a minimal Cypress test that verifies the Documentation card fetches and displays a file using the existing backend.

Which of the three would you like me to do next?

