param(
    [Parameter(Mandatory = $true)]
    [string]$CondaHook,

    [Parameter(Mandatory = $true)]
    [string]$CondaPath,

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
conda activate $CondaPath

if (-not (Get-Command streamlit -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] streamlit was not found after activating Conda."
    Write-Host "Run: pip install -r requirements.txt"
    exit 1
}

streamlit run whisper_app.py --server.maxUploadSize $UploadLimitMb
exit $LASTEXITCODE
