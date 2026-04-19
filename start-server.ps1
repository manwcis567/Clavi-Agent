$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Starting Clavi Agent server on http://127.0.0.1:8000"
uv run python -m uvicorn clavi_agent.server:app --host 127.0.0.1 --port 8000


