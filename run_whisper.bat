@echo off
setlocal

title Whisper AI Launcher

rem Keep this file ASCII-only. Windows cmd.exe can misread UTF-8 .bat files.
cd /d "%~dp0"

set "UPLOAD_LIMIT_MB=5000"

set "CONDA_PATH=%USERPROFILE%\anaconda3"
if not exist "%CONDA_PATH%\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\anaconda3"
)
if not exist "%CONDA_PATH%\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\miniconda3"
)

echo [1/3] Checking project folder...
echo Current folder: %cd%

if not exist "whisper_app.py" (
    echo [ERROR] whisper_app.py was not found in this folder.
    echo Put run_whisper.bat and whisper_app.py in the same folder.
    pause
    exit /b 1
)

set "PS_HOOK=%CONDA_PATH%\shell\condabin\conda-hook.ps1"
if not exist "%PS_HOOK%" (
    echo [ERROR] Conda PowerShell hook was not found.
    echo Expected: %PS_HOOK%
    echo Edit CONDA_PATH in run_whisper.bat if Conda is installed elsewhere.
    pause
    exit /b 1
)

if not exist "start_whisper.ps1" (
    echo [ERROR] start_whisper.ps1 was not found in this folder.
    pause
    exit /b 1
)

echo [2/3] Activating Conda...
echo [3/3] Starting Streamlit...

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_whisper.ps1" -CondaHook "%PS_HOOK%" -CondaPath "%CONDA_PATH%" -UploadLimitMb %UPLOAD_LIMIT_MB%
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
