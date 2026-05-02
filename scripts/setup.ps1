param(
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== Badminton Analysis setup =="

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.10 or newer, then run this script again."
}

$pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Python: $pythonVersion"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$PipExe = Join-Path $Root ".venv\Scripts\pip.exe"

& $PythonExe -m pip install --upgrade pip
& $PipExe install -r requirements.txt

New-Item -ItemType Directory -Force -Path "models", "data\raw", "data\output", "data\logs", "data\calibration" | Out-Null

if (-not $SkipModelDownload) {
    Write-Host "Downloading default public model assets..."
    & $PythonExe "scripts\download_assets.py"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Put your input videos under data\raw"
Write-Host "  2. Run: python src\add_source.py"
Write-Host "  3. Run: python src\main.py --source <source_name> --output data\output\result.mp4 --duration 30"
