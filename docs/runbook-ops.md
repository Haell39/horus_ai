# Runbook Ops — Horus AI (passos rápidos)

Resumo rápido: comandos PowerShell copy/paste para iniciar/parar serviços, checar logs e recuperar o HLS.

1. Iniciar backend (dev)

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # só se necessário
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. Iniciar frontend (dev)

```powershell
cd frontend
npm install
npm start
```

3. Start/Stop ingest (SRT → HLS)

```powershell
# Start (substitua a URL SRT real)
$body = @{ url = 'srt://MEU.SRT.ENDPOINT:PORT?mode=caller&passphrase=...' ; fps = 1.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body $body -ContentType 'application/json'

# Stop
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/stop

# Status
Invoke-RestMethod -Method Get -Uri http://localhost:8000/api/v1/streams/status
```

4. Verificar playlist e logs HLS

```powershell
Get-Content backend\static\hls\stream.m3u8 -Raw
Get-Content backend\static\hls\hls_ffmpeg.log -Tail 200
```

5. Matar ffmpeg pendente

```powershell
Get-Process -Name ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

6. Limpar HLS manualmente (se preciso)

```powershell
Remove-Item -Recurse -Force backend\static\hls\*
# recria um minimal playlist
Set-Content -Path backend\static\hls\stream.m3u8 -Value "#EXTM3U`n#EXT-X-VERSION:3`n#EXT-X-PLAYLIST-TYPE:VOD`n#EXT-X-ENDLIST`n"
```

7. Dicas rápidas

- NÃO use `--reload` no uvicorn ao testar streams (reinicia processos e mata ffmpeg).
- Se o player não reproduz, confira `stream.m3u8` e `hls_ffmpeg.log`.
- Para debugging de inferência, verifique os logs de startup (carregamento dos modelos) no console do backend.
