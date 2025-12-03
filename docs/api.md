# API — Endpoints do Horus AI

Base: `http://<backend-host>:8000`

---

## 1. Stream Control

### Start Ingest (SRT → HLS)

- **POST** `/api/v1/streams/start`
- Body (JSON): `{ "url": "srt://...", "fps": 1.0 }`

```powershell
$body = @{ url = 'srt://SEU.SRT.ENDPOINT:PORT?mode=caller&passphrase=...' ; fps = 1.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body $body -ContentType 'application/json'
```

### Stop Ingest

- **POST** `/api/v1/streams/stop`

### Status

- **GET** `/api/v1/streams/status`
- Response: `{ "running": true, "fps": 1.0 }`

---

## 2. Ocorrências

### Listar

- **GET** `/api/v1/ocorrencias`
- Response: Lista JSON de ocorrências (id, start_ts, end_ts, duration_s, category, type, severity, confidence, evidence)

### Criar (interno)

- **POST** `/api/v1/ocorrencias`

---

## 3. WebSocket (Realtime)

- **URL**: `ws://<backend-host>:8000/ws/ocorrencias`
- Broadcast para novas ocorrências
- Mensagem: `{ "type": "nova_ocorrencia", "data": { ... } }`

---

## 4. Análise Manual

### Upload de Vídeo

- **POST** `/api/v1/analysis/upload`
- Multipart form com arquivo de vídeo

---

## 5. Informações ML

- **GET** `/api/v1/ml/info` - Status dos modelos carregados

---

## 6. Arquivos Estáticos

| Recurso      | URL                                                |
| ------------ | -------------------------------------------------- |
| HLS playlist | `http://<backend-host>:8000/hls/stream.m3u8`       |
| Clips        | `http://<backend-host>:8000/clips/<clip_name>.mp4` |

---

## 7. Admin

- **POST** `/api/v1/admin/clear` - Limpar dados
