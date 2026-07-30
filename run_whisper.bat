@echo off
setlocal

title Whisper AI Launcher

rem Keep this file ASCII-only. Windows cmd.exe can misread UTF-8 .bat files.
cd /d "%~dp0"

set "UPLOAD_LIMIT_MB=5000"

rem Which address to listen on.
rem   tailscale = only reachable over Tailscale (recommended; keeps it off the LAN)
rem   <empty>   = listen on every interface, anyone on the local network can open it
rem   <an IP>   = listen on that address only
set "BIND_ADDRESS=tailscale"

set "CONDA_PATH="

rem Prefer the Conda that is already available on PATH.
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined CONDA_PATH (
    for %%J in ("%%~dpI..") do set "CONDA_PATH=%%~fJ"
)

rem Fall back to common installation folders.
if not defined CONDA_PATH if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_PATH=%USERPROFILE%\anaconda3"
if not defined CONDA_PATH if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" set "CONDA_PATH=C:\ProgramData\anaconda3"
if not defined CONDA_PATH if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_PATH=%USERPROFILE%\miniconda3"

set "CONDA_ENV=%WHISPER_CONDA_ENV%"
if not defined CONDA_ENV set "CONDA_ENV=%CONDA_PATH%"

echo [1/3] Checking project folder...
echo Current folder: %cd%

if not exist "whisper_app.py" (
    echo [ERROR] whisper_app.py was not found in this folder.
    echo Put run_whisper.bat and whisper_app.py in the same folder.
    pause
    exit /b 1
)

if not defined CONDA_PATH (
    echo [ERROR] Conda was not found on PATH or in a common install folder.
    echo Open Anaconda Prompt and run this file again, or add Conda to PATH.
    pause
    exit /b 1
)

set "PS_HOOK=%CONDA_PATH%\shell\condabin\conda-hook.ps1"
if not exist "%PS_HOOK%" (
    echo [ERROR] Conda PowerShell hook was not found.
    echo Expected: %PS_HOOK%
    echo Conda detected at: %CONDA_PATH%
    pause
    exit /b 1
)

if not exist "start_whisper.ps1" (
    echo [ERROR] start_whisper.ps1 was not found in this folder.
    pause
    exit /b 1
)

echo [2/3] Activating Conda...
echo Conda folder: %CONDA_PATH%
echo Conda environment: %CONDA_ENV%
echo [3/3] Starting Streamlit...

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_whisper.ps1" -CondaHook "%PS_HOOK%" -CondaEnvironment "%CONDA_ENV%" -UploadLimitMb %UPLOAD_LIMIT_MB% -BindAddress "%BIND_ADDRESS%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Startup failed.
    echo If packages are missing, run this in your Conda environment:
    echo pip install -r requirements.txt
    echo.
)

pause
exit /b %EXIT_CODE%
