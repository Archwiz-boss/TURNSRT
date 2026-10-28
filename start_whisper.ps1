param(
    [Parameter(Mandatory = $true)]
    [string]$CondaHook,

    [Parameter(Mandatory = $true)]
    [string]$CondaEnvironment,

    [int]$UploadLimitMb = 5000,

    # "" listens on every interface, "tailscale" resolves this machine's Tailscale
    # address, or pass a literal IP. Binding to one address is what keeps the app off
    # the local network.
    [string]$BindAddress = ""
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
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
conda activate $CondaEnvironment 2>$null
$activationExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference

if ($activationExitCode -ne 0) {
    Write-Host "[ERROR] Conda environment activation failed: $CondaEnvironment"
    exit 1
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python was not found after activating Conda."
    exit 1
}

$bootstrapPath = Join-Path -Path $PSScriptRoot -ChildPath "streamlit_bootstrap.py"
if (-not (Test-Path -LiteralPath $bootstrapPath)) {
    Write-Host "[ERROR] streamlit_bootstrap.py was not found in this folder."
    exit 1
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or
    -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] FFmpeg or FFprobe was not found on PATH."
    Write-Host "Install FFmpeg and make sure both commands work in a terminal."
    exit 1
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python $bootstrapPath --check
$dependencyExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($dependencyExitCode -ne 0) {
    Write-Host "[ERROR] One or more Python packages are missing."
    Write-Host "Run: pip install -r requirements.txt"
    exit 1
}

if ($BindAddress -eq "tailscale") {
    $tailscaleIp = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.InterfaceAlias -like "*Tailscale*" } |
        Select-Object -First 1
    if (-not $tailscaleIp) {
        Write-Host "[WARN] No Tailscale interface was found. Falling back to localhost."
        Write-Host "Only this computer will be able to open the app."
        $BindAddress = "127.0.0.1"
    }
    else {
        $BindAddress = $tailscaleIp.IPAddress
    }
}

$streamlitArgs = @("run", "whisper_app.py", "--server.maxUploadSize", $UploadLimitMb)
if ($BindAddress) {
    $streamlitArgs += @("--server.address", $BindAddress)
    Write-Host ""
    Write-Host "Listening on $BindAddress only - the local network cannot reach this."
    if ($BindAddress -eq "127.0.0.1") {
        Write-Host "Open http://127.0.0.1:8501"
    }
    else {
        Write-Host "Open http://${BindAddress}:8501 (works on this machine too; localhost will not)."
    }
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "Listening on all interfaces - anyone on your local network can open this."
    Write-Host ""
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
python $bootstrapPath @streamlitArgs
$streamlitExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
exit $streamlitExitCode
