# API — Endpoints úteis (exemplos)

Base: `http://<backend-host>:8000`

1. Start ingest (SRT → HLS)

- POST `/api/v1/streams/start`
- Body (JSON): `{ "url": "srt://...", "fps": 1.0 }`

PowerShell example:

```powershell
$body = @{ url = 'srt://MEU.SRT.ENDPOINT:PORT?mode=caller&passphrase=...' ; fps = 1.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body $body -ContentType 'application/json'
```

2. Stop ingest

- POST `/api/v1/streams/stop`

3. Status

- GET `/api/v1/streams/status` -> `{ "running": true, "fps": 1.0 }`

4. Ocorrências

- GET `/api/v1/ocorrencias` -> lista JSON de ocorrências (id, start_ts, end_ts, duration_s, category, type, severity, confidence, evidence)
- POST `/api/v1/ocorrencias` -> cria ocorrência (usado internamente)

5. WebSocket (realtime)

- `ws://<backend-host>:8000/ws/ocorrencias` — broadcast para novas ocorrências (mensagem: { type: 'nova_ocorrencia', data: { ... } })

6. HLS & Clips

- HLS playlist: `http://<backend-host>:8000/hls/stream.m3u8`
- Clips: `http://<backend-host>:8000/clips/<clip_name>.mp4`

7. Observações

- Start retorna sucesso somente quando playlist ficou pronta (evita attach race no frontend).
- Não exponha passphrases/segredos em chamadas que vão para o frontend; envie apenas pelo backend via `.env` ou vault.
