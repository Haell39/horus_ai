# run_setup.ps1
# One-time setup script: creates virtualenv for backend and installs Python requirements,
# and runs npm install in frontend. Run this once (double-click or Run with PowerShell).

$repoRoot = "$PSScriptRoot"
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"

Write-Output "Running one-time setup..."

# Backend venv and pip install
if (-Not (Test-Path (Join-Path $backendDir ".venv"))) {
    Write-Output "Creating virtualenv in $backendDir\.venv ..."
    python -m venv (Join-Path $backendDir ".venv")
} else {
    Write-Output "Virtualenv already exists."
}

Write-Output "Installing backend requirements... (this may take a few minutes)"
Push-Location $backendDir
if (Test-Path ".venv") { . .\.venv\Scripts\Activate.ps1 }
pip install --upgrade pip
if (Test-Path "requirements.txt") { pip install -r requirements.txt }
Pop-Location

# Frontend npm install
Write-Output "Installing frontend dependencies... (this may take a few minutes)"
Push-Location $frontendDir
if (Test-Path "package.json") { npm install }
Pop-Location

Write-Output "Setup complete. You can now run run_all.ps1 to start backend and frontend."
