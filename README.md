# Horus AI

## Resumo

Horus AI é uma aplicação de monitoramento em tempo real para ingestão de streams SRT, geração de HLS para visualização no navegador, execução de inferência leve com modelos TFLite e persistência de ocorrências em banco de dados. O projeto contém backend (FastAPI) e frontend (Angular).

## Estrutura principal

- `backend/` — FastAPI app, ML (TFLite), ingest de stream (ffmpeg), APIs e mount de arquivos estáticos.

  - `backend/app/streams/srt_reader.py` — controladora de ingest SRT → HLS + extractor de frames para inferência.
  - `backend/app/api/endpoints/streams.py` — endpoints start/stop/status para ingest.
  - `backend/app/api/endpoints/ws.py` — WebSocket para broadcast de ocorrências.
  - `backend/app/ml/` — wrappers para carregar e executar modelos TFLite.
  - `backend/static/hls/` — playlist e segmentos HLS gerados (ignorado pelo git).
  - `backend/static/clips/` — clips gerados a partir de frames.

- `frontend/` — Angular app, contém a página `Monitoramento` que se conecta ao HLS servido pelo backend e ao WebSocket.

## Requisitos

- Windows ou Linux
- Python 3.11+
- Node.js + npm
- ffmpeg disponível no PATH (necessário para SRT/HLS e geração de frames/clips)
- Postgres (opcional, para persistência de ocorrências) — configurar `DATABASE_URL` no `.env`

## Quick Start (desenvolvimento) — PowerShell

1. Backend

```powershell
cd backend
# criar e ativar venv (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# inicie o servidor sem --reload durante testes de streaming (evita reinícios que fecham ffmpeg/WS)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Frontend

```powershell
cd frontend
npm install
npm start   # roda o ng serve (dev server)
```

3. Testar endpoints

- Health:

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:8000/
```

- Iniciar ingest (SRT → HLS):

```powershell
# substitua a URL SRT real
$body = @{ url = 'srt://MEU.SRT.ENDPOINT:PORT?mode=caller&passphrase=...' ; fps = 1.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body $body -ContentType 'application/json'
```

- Parar ingest:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/stop
```

- Status:

```powershell
Invoke-RestMethod -Method Get -Uri http://localhost:8000/api/v1/streams/status
```

## Acesso ao player e WS

- HLS: `http://<backend-host>:8000/hls/stream.m3u8`
- Clips: `http://<backend-host>:8000/clips/<clip_name>.mp4`
- WebSocket para ocorrências: `ws://<backend-host>:8000/ws/ocorrencias`

## Dicas de desenvolvimento & troubleshooting

- NÃO use `--reload` no uvicorn enquanto estiver testando ingest de streams. O reload reinicia o processo Python, fechando WebSockets e os subprocessos ffmpeg em execução.
- Se os segmentos `.ts` ou o `stream.m3u8` não aparecem, verifique `backend/static/hls/hls_ffmpeg.log` para mensagens do ffmpeg.
- Mensagens comuns do ffmpeg que indicam problemas do stream remoto:
  - `ERROR:BACKLOG` — o servidor SRT remoto rejeitou a handshake (backlog cheio).
  - `Missing reference picture`, `sps_id 0 out of range`, `non-existing PPS` — problemas nos NALs H.264 vindos do emissor; podem causar frames perdidos.
- Para inspecionar rapidamente (PowerShell):

```powershell
Get-ChildItem backend\static\hls\
Get-Content backend\static\hls\hls_ffmpeg.log -Tail 200
Get-Content backend\static\hls\stream.m3u8 -Raw
Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime
```

- Para matar ffmpeg(s) pendentes:

```powershell
Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

## Segurança e produção

- Em produção mova credenciais e passphrases SRT para variáveis de ambiente seguras / Vault e NÃO as exponha no frontend.
- Restrinja `CORS` em `backend/app/main.py` para os domínios do frontend em produção (não use `*`).
- Considere usar um process manager (systemd, docker, or supervisor) para garantir ffmpeg e o backend iniciem, e logrotate para `hls_ffmpeg.log`.

## Próximos passos recomendados

- Adicionar endpoint `/api/v1/streams/cleanup` para kill + cleanup remoto (útil para recuperação manual).
- Implementar retry/backoff no `SRTIngestor.start()` para lidar com rejeições SRT transientes.
- Adicionar testes de integração leve que simulam um SRT/HLS small stream e validam o fluxo `/streams/start` -> `stream.m3u8` disponível.
- Adicionar documentação de runbook (DevOps) com checks e comandos de recuperação rápida.

## Contato / Contribuição

Abra issues e PRs no repositório. Se for contribuir com mudanças em ffmpeg/startup behavior, teste localmente sem `--reload` e valide que o frontend consegue se conectar automaticamente depois do `start`.
