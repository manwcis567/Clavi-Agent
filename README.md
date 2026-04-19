# Clavi Agent

> The key to next-gen intelligent action

**Clavi Agent** is an evolved agent project that grew out of earlier mini-agent exploration and now follows its own roadmap while moving alongside the broader mini-agent ecosystem. It keeps a lightweight, professional runtime, adds durable memory and account isolation, and stays flexible about which LLM provider or model you connect.

This public snapshot intentionally omits internal design notes, prompt markdown, and other non-`README.md` documentation files.

This project comes packed with features designed for a robust and intelligent agent development experience:

*   ✅ **Full Agent Execution Loop**: A complete and reliable foundation with a basic toolset for file system and shell operations.
*   ✅ **Persistent Memory**: An active **Session Note Tool** ensures the agent retains key information across multiple sessions.
*   ✅ **Intelligent Context Management**: Automatically summarizes conversation history to handle contexts up to a configurable token limit, enabling infinitely long tasks.
*   ✅ **Claude Skills Support**: Supports Claude-style skill loading and skill-aware workflows.
*   ✅ **MCP Tool Integration**: Natively supports MCP for tools like knowledge graph access and web search.
*   ✅ **Comprehensive Logging**: Detailed logs for every request, response, and tool execution for easy debugging.
*   ✅ **Dynamic Agent Studio**: Use the built-in Marketplace to visually configure, save, and deploy specialized Agents directly into chat sessions.
*   ✅ **Clean & Simple Design**: A beautiful CLI and web interface codebase that is easy to understand, making it a strong starting point for building advanced agents.

## Table of Contents

- [Clavi Agent](#clavi-agent)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [1. Prepare Model Access](#1-prepare-model-access)
    - [2. Choose Your Usage Mode](#2-choose-your-usage-mode)
      - [🚀 Quick Start Mode (Recommended for Beginners)](#-quick-start-mode-recommended-for-beginners)
      - [🔧 Development Mode](#-development-mode)
    - [Memory Rollout Flags](#memory-rollout-flags)
  - [Local Accounts](#local-accounts)
    - [Root Bootstrap](#root-bootstrap)
    - [Upgrade Existing Data](#upgrade-existing-data)
  - [Usage Examples](#usage-examples)
    - [Task Execution](#task-execution)
    - [Using a Claude Skill (e.g., PDF Generation)](#using-a-claude-skill-eg-pdf-generation)
    - [Web Search \& Summarization (MCP Tool)](#web-search--summarization-mcp-tool)
  - [Deliverable-First Workflow](#deliverable-first-workflow)
    - [How File Uploads Work](#how-file-uploads-work)
    - [Upload Limits And Safety](#upload-limits-and-safety)
    - [Example Flows](#example-flows)
  - [Testing](#testing)
    - [Quick Run](#quick-run)
    - [Test Coverage](#test-coverage)
  - [Troubleshooting](#troubleshooting)
    - [SSL Certificate Error](#ssl-certificate-error)
    - [Module Not Found Error](#module-not-found-error)
  - [Contributing](#contributing)
  - [License](#license)
  - [References](#references)

## Quick Start

### 1. Prepare Model Access

Clavi Agent documentation is no longer tied to a single model vendor. Before your first run, prepare the three values below from the LLM service you plan to use:

- An API key
- The matching API base URL
- The model name

Common choices include OpenAI-compatible and Anthropic-compatible endpoints. As long as your deployment matches the client path you configure, Clavi Agent can be pointed at the provider you already use.

**Recommended preparation steps:**
1. Create or copy an API key from your provider console.
2. Confirm the base URL required by that provider.
3. Pick the model name you want Clavi Agent to call.
4. Keep those values ready for `config.yaml`.

### 2. Choose Your Usage Mode

**Prerequisites: Install uv**

Both usage modes require uv. If you don't have it installed:

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
python -m pip install --user pipx
python -m pipx ensurepath
# Restart PowerShell after installation

# After installation, restart your terminal or run:
source ~/.bashrc  # or ~/.zshrc (macOS/Linux)
```

We offer two usage modes. Choose based on whether you want to try the runtime quickly or work on the codebase directly.

#### 🚀 Quick Start Mode (Recommended for Beginners)

Perfect for users who want to quickly try Clavi Agent without setting up an editable development environment.

**Installation:**

```bash
# Install directly from the current repository
uv tool install git+https://github.com/manwcis567/Clavi-Agent.git
```

**Configuration:**

Create the runtime config file in `~/.clavi-agent/config/config.yaml`, then fill in your provider details:

```yaml
api_key: "YOUR_API_KEY_HERE"
api_base: "https://your-llm-endpoint/v1"
model: "your-model-name"
```

If you need runtime helpers such as `node`, `npm`, or `clawhub`, run:

```bash
clavi-agent-setup-runtime --config ~/.clavi-agent/config/config.yaml
```

> 💡 **Tip**: The package and CLI name remain `clavi-agent`, while the project itself continues to evolve alongside mini-agent.

**Start Using:**

```bash
clavi-agent                                    # Use current directory as workspace
clavi-agent --workspace /path/to/your/project  # Specify workspace directory
clavi-agent --version                          # Check version

# Management commands
uv tool upgrade clavi-agent                    # Upgrade to latest version
uv tool uninstall clavi-agent                  # Uninstall if needed
uv tool list                                   # View all installed tools
```

#### 🔧 Development Mode

For developers who need to modify code, add features, or debug.

**Installation & Configuration:**

```bash
# 1. Clone the repository
git clone https://github.com/manwcis567/Clavi-Agent.git ClaviAgent
cd ClaviAgent

# 2. Install uv (if you haven't)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex
# Restart terminal after installation

# 3. Sync dependencies
uv sync

# Alternative: Install dependencies manually (if not using uv)
# pip install -r requirements.txt
# Or install required packages:
# pip install tiktoken pyyaml httpx pydantic requests prompt-toolkit mcp

# 4. Initialize Claude Skills (Optional)
git submodule update --init --recursive

# 5. Copy config template
```

**macOS/Linux:**
```bash
cp clavi_agent/config/config-example.yaml clavi_agent/config/config.yaml
```

**Windows:**
```powershell
Copy-Item clavi_agent\config\config-example.yaml clavi_agent\config\config.yaml

# 6. Edit config file
vim clavi_agent/config/config.yaml  # Or use your preferred editor
```

**Install runtime dependencies before first run:**
```bash
uv run clavi-agent-setup-runtime --config clavi_agent/config/config.yaml
```

Fill in your API credentials and provider endpoint:

```yaml
api_key: "YOUR_API_KEY_HERE"
api_base: "https://your-llm-endpoint/v1"
model: "your-model-name"
max_steps: 100
workspace_dir: "./workspace"
```

> 📖 Full configuration guide: See [config-example.yaml](clavi_agent/config/config-example.yaml)

### Memory Rollout Flags

Long-term memory can be rolled out independently from the base session and agent APIs:

```yaml
feature_flags:
  enable_compact_prompt_memory: true
  enable_session_retrieval: true
  enable_learned_workflow_generation: true
  enable_external_memory_providers: true

memory_provider:
  provider: "local"
```

- `enable_compact_prompt_memory`: injects `User Profile Summary`, `Stable Working Preferences`, and workspace-local `Relevant Agent Memory` blocks.
- `enable_session_retrieval`: enables `/api/session-history`, the `search_session_history` tool, and `retrieved_context` prompt injection.
- `enable_learned_workflow_generation`: auto-captures learned workflow candidates from successful runs.
- `enable_external_memory_providers`: allows non-local memory providers such as the MCP memory adapter. If this flag is disabled while `provider: "mcp"` is configured, Clavi Agent falls back to the local SQLite provider.

Compatibility rules:

- Existing session and agent APIs continue to work when these flags are disabled; only the memory-specific surfaces are gated.
- Existing `.agent_memory.json` files remain workspace-local `agent_memory` storage. They are still read and written with UTF-8, and are not auto-migrated into user-level memory.
- When browser auth or an explicit account id is unavailable, the runtime falls back to the built-in `root` account so memory and retrieval still have a stable local identity.

**Run Methods:**

Choose your preferred run method:

```bash
# Method 1: Run as module directly (good for debugging)
uv run python -m clavi_agent.cli

# Method 2: Install in editable mode (recommended)
uv tool install -e .
# After installation, run from anywhere and code changes take effect immediately
clavi-agent
clavi-agent --workspace /path/to/your/project
```

**Web UI / Agent Studio:**

```bash
# Cross-platform entry point
uv run clavi-agent-server

# Convenience scripts from the repository root
bash start-server.sh   # macOS/Linux
./start-server.ps1     # PowerShell
start-server.cmd       # Windows CMD
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

> This public snapshot omits the extended Markdown guides that exist in the private working repository.

### Agent Studio Template Policy

The Marketplace form now separates the "identity prompt layer" from the "global delegation policy layer":

- `System Prompt` still defines the agent's role, tone, and domain responsibilities.
- `Delegation Policy` defines supervisor / worker boundaries such as `hybrid`, `prefer_delegate`, and `supervisor_only`, plus whether writes, shell actions, or stateful MCP calls must be delegated.
- `LLM Routing` lets you override planner and worker model profiles independently. Leaving fields blank inherits the active account runtime defaults.

System templates are shown in a read-only view so the current supervisor strategy is visible without letting those defaults drift. Custom templates can edit the same policy fields directly instead of copying a large delegation block back into `System Prompt`.

## Local Accounts

Clavi Agent `V1` now includes a local account system for the Web UI and protected APIs.

- Browser-facing management APIs use cookie-backed sessions.
- A built-in `root` account is seeded automatically on startup.
- Public self-registration is disabled by default; additional accounts must be created by `root`.
- Historical single-user data can be migrated into `root` ownership without moving existing workspace files first.

### Root Bootstrap

On startup, the server initializes the `root` account from the auth config:

```yaml
auth:
  auto_seed_root: true
  root_username: "root"
  root_display_name: "Root"
  root_password: "ChangeMe123!"
  web_session_ttl_hours: 12
```

You can also provide the password through `CLAVI_AGENT_ROOT_PASSWORD` (or the configured `root_password_env`) instead of storing it in the file. If no password source is configured, Clavi Agent generates a one-time temporary password and prints it in the startup log.

### Upgrade Existing Data

When upgrading an older single-user workspace, run the built-in migration before asking other users to log in:

```bash
uv run clavi-agent-migrate-root-data --config clavi_agent/config/config.yaml
```

The migration command:

- Backs up `session_db` and `agent_db` first
- Creates the `root` account if needed
- Backfills historical records to `root`
- Prints a per-table report showing total rows, backfilled rows, and remaining empty ownership rows

## Usage Examples

Here are a few examples of what Clavi Agent can do.

### Task Execution

*In this demo, the agent is asked to create a simple, beautiful webpage and display it in the browser, showcasing the basic tool-use loop.*

![Demo GIF 1: Basic Task Execution](docs/assets/demo1-task-execution.gif "Basic Task Execution Demo")

### Using a Claude Skill (e.g., PDF Generation)

*Here, the agent leverages a Claude Skill to create a professional document (like a PDF or DOCX) based on the user's request, demonstrating its advanced capabilities.*

![Demo GIF 2: Claude Skill Usage](docs/assets/demo2-claude-skill.gif "Claude Skill Usage Demo")

### Web Search & Summarization (MCP Tool)

*This demo shows the agent using its web search tool to find up-to-date information online and summarize it for the user.*

![Demo GIF 3: Web Search](docs/assets/demo3-web-search.gif "Web Search Demo")

## Deliverable-First Workflow

Clavi Agent now supports a deliverable-first workflow in the Web UI:

1. Upload one or more source files into a session.
2. Attach them to the next chat turn.
3. Let the agent inspect the source and create a revised copy or a new export.
4. Open the run detail view to preview, open, or download the final deliverable and supporting artifacts.

### How File Uploads Work

- Uploaded files are stored under `.clavi_agent/uploads/<session_id>/<upload_id>/<safe_name>` inside the session workspace, so normal file tools can read and revise them.
- The browser uploads files first, then sends `attachment_ids` with the next chat request. This keeps file transfer separate from the SSE chat stream.
- Revised outputs and generated reports are promoted into run deliverables, so the run detail panel highlights the primary result instead of only showing raw artifact paths.

### Upload Limits And Safety

- Maximum upload size is `25 MB` per file.
- Supported formats include `Markdown`, plain text, `JSON`, `CSV` / `TSV`, `HTML`, `XML`, `YAML`, images (`PNG`, `JPG`, `GIF`, `SVG`, `WEBP`), `PDF`, and Office documents (`DOC`, `DOCX`, `PPT`, `PPTX`, `XLS`, `XLSX`, `RTF`).
- Executable or script-like uploads such as `EXE`, `DLL`, `JS`, `PY`, `SH`, `BAT`, and `PS1` are blocked.
- Uploaded files default to copy-on-write revision. Unless the user explicitly asks to overwrite the original, the agent should create a revised copy in the same upload directory.
- If a tool targets the original uploaded file itself, Clavi Agent now raises that action as a higher-risk approval request by default unless the template has an explicit auto-approve rule for the tool.

### Example Flows

- Upload a `draft.md` file and ask: `Please tighten the structure, keep the tone, and save a revised copy.`
- Upload a `review.docx` file and ask: `Review this document, keep the revised DOCX, and export a clean PDF deliverable.`
- Ask: `Create a weekly status report for this repository and return the final result as Markdown or DOCX.`

In the completed run view, uploaded files, revised files, and final deliverables each expose `Preview`, `Open`, and `Download` actions when the format supports them.

## Testing

The project includes comprehensive test cases covering unit tests, functional tests, and integration tests.

### Quick Run

```bash
# Run all tests
pytest tests/ -v

# Run core functionality tests
pytest tests/test_agent.py tests/test_note_tool.py -v
```

### Test Coverage

- ✅ **Unit Tests** - Tool classes, LLM client
- ✅ **Functional Tests** - Session Note Tool, MCP loading
- ✅ **Integration Tests** - Agent end-to-end execution
- ✅ **External Services** - Git MCP Server loading
- ✅ **Account System** - login/logout, session expiry, root migration, cross-account access control, webhook account routing

## Troubleshooting

### SSL Certificate Error

If you encounter `[SSL: CERTIFICATE_VERIFY_FAILED]` error:

**Quick fix for testing** (modify `clavi_agent/llm.py`):
```python
# Line 50: Add verify=False to AsyncClient
async with httpx.AsyncClient(timeout=120.0, verify=False) as client:
```

**Production solution**:
```bash
# Update certificates
pip install --upgrade certifi

# Or configure system proxy/certificates
```

### Module Not Found Error

Make sure you're running from the project directory:
```bash
cd ClaviAgent
python -m clavi_agent.cli
```

## Contributing

Issues and Pull Requests are welcome!

Please open an issue or pull request if you want to discuss improvements.

## License

This project is licensed under the [MIT License](LICENSE).

## References

- ClaviAgent repository: https://github.com/manwcis567/Clavi-Agent.git
- Anthropic API: https://docs.anthropic.com/claude/reference
- Claude Skills: https://github.com/anthropics/skills
- MCP Servers: https://github.com/modelcontextprotocol/servers

---

**⭐ If this project helps you, please give it a Star!**
