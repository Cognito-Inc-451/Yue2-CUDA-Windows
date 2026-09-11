# YuE2-3B Music Generator — one-click setup (Windows)
#
# Run from the project root:
#   .\setup.ps1
#
# Requires: Python 3.11 on PATH, and ffmpeg on PATH (for MP3 encoding).
# After it finishes, start the app with:
#   .venv\Scripts\python app.py
# → http://127.0.0.1:7860

$ErrorActionPreference = "Stop"

Write-Host "==> Creating virtual environment (.venv)..." -ForegroundColor Cyan
python -m venv .venv

$py = ".venv\Scripts\python"

Write-Host "==> Upgrading pip..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip

Write-Host "==> Installing CUDA torch (cu128) FIRST..." -ForegroundColor Cyan
& $py -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

Write-Host "==> Installing pre-requirements..." -ForegroundColor Cyan
& $py -m pip install -r pre-requirements.txt

Write-Host "==> Installing requirements..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

Write-Host "==> Applying Windows patches (idempotent)..." -ForegroundColor Cyan
& $py apply_patches.py

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Start the app with:" -ForegroundColor Green
Write-Host "    .venv\Scripts\python app.py" -ForegroundColor Green
Write-Host "Then open http://127.0.0.1:7860" -ForegroundColor Green
