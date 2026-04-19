@echo off
setlocal

cd /d "%~dp0"

echo Starting Clavi Agent server on http://127.0.0.1:8000
uv run python -m uvicorn clavi_agent.server:app --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo Server exited with an error.
    pause
)


