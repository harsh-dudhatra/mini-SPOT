@echo off
rem ============================================================================
rem Mini-SPOT - setup_windows.bat
rem Sets up the Python env (.venv) on Windows with uv. NVIDIA GPU is used if found.
rem
rem   setup_windows.bat            auto: CUDA 12.8 torch if nvidia-smi works, else CPU
rem   set TORCH=cpu ^& setup_windows.bat     force CPU build (or TORCH=cu128)
rem
rem Afterwards:  .venv\Scripts\activate
rem ============================================================================
setlocal
cd /d "%~dp0"

rem ---- 1. uv (installs itself into %USERPROFILE%\.local\bin) ----------------
where uv >nul 2>&1
if errorlevel 1 (
    echo ^>^> uv not found, installing...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>&1
if errorlevel 1 (
    echo !! uv install failed. Install it manually: https://docs.astral.sh/uv/
    exit /b 1
)

rem ---- 2. Pick the torch build: CUDA if an NVIDIA driver is present ----------
if not defined TORCH (
    nvidia-smi >nul 2>&1
    if errorlevel 1 (set "TORCH=cpu") else (set "TORCH=cu128")
)
echo ^>^> torch build: %TORCH%

rem ---- 3. Create / update .venv (uv downloads Python 3.11 itself) -----------
uv sync --extra %TORCH%
if errorlevel 1 (
    echo !! uv sync failed
    exit /b 1
)

rem ---- 4. Sanity check -------------------------------------------------------
uv run --no-sync python -c "import torch, mujoco, stable_baselines3 as sb3; print('>> mujoco', mujoco.__version__, '| sb3', sb3.__version__, '| torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if errorlevel 1 exit /b 1

echo.
echo ^>^> Env ready. Activate with:  .venv\Scripts\activate
