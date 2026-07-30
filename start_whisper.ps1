param(
    [Parameter(Mandatory = $true)]
    [string]$CondaHook,

    [Parameter(Mandatory = $true)]
    [string]$CondaEnvironment,

    [int]$UploadLimitMb = 5000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath "whisper_app.py")) {
    Write-Host "[ERROR] whisper_app.py was not found in this folder."
    exit 1
}

if (-not (Test-Path -LiteralPath $CondaHook)) {
    Write-Host "[ERROR] Conda PowerShell hook was not found: $CondaHook"
    exit 1
}

. $CondaHook
conda activate $CondaEnvironment

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python was not found after activating Conda."
    exit 1
}

if (-not (Get-Command streamlit -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] streamlit was not found after activating Conda."
    Write-Host "Run: pip install -r requirements.txt"
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
    -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] FFmpeg or FFprobe was not found on PATH."
    Write-Host "Install FFmpeg and make sure both commands work in a terminal."
    exit 1
}

python -c "import numpy, opencc, plotly, streamlit, torch, transformers, whisper" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] One or more Python packages are missing."
    Write-Host "Run: pip install -r requirements.txt"
    exit 1
}

streamlit run whisper_app.py --server.maxUploadSize $UploadLimitMb
exit $LASTEXITCODE
