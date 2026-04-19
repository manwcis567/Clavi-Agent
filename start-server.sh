#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Clavi Agent server on http://127.0.0.1:8000"
exec uv run python -m uvicorn clavi_agent.server:app --host 127.0.0.1 --port 8000


