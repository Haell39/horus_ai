# run_all.ps1
# Opens two PowerShell windows: one for the backend (uvicorn) and one for the frontend (npm start)
# Usage: double-click this file in Explorer or right-click -> Run with PowerShell

# Adjust these paths if your repo is in a different location
$repoRoot = "$PSScriptRoot"
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

# Backend command: activate venv (if present) and start uvicorn
$backendCmd = @"
cd `"$backendDir`"
if (Test-Path .venv) { . .\.venv\Scripts\Activate.ps1 }
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
"@

# Frontend command: npm start
$frontendCmd = @"
cd `"$frontendDir`"
npm start
"@

Write-Output "Starting backend and frontend in separate windows..."

# Start backend window
Start-Process -FilePath powershell -ArgumentList "-NoExit","-Command $backendCmd"

# Start frontend window
Start-Process -FilePath powershell -ArgumentList "-NoExit","-Command $frontendCmd"

Write-Output "Done. Two windows were started. Wait for the backend log 'Uvicorn running' and the frontend dev server to be ready (http://localhost:4200)."
